"""Replicate https://github.com/davidcpage/cifar10-fast in JAX.

Some notes:
* Entirety of the CIFAR-10 dataset is loaded into GPU memory, for speeeedz.
* Training epochs are fully jit'd with `lax.scan`.
* Like the pytorch version, we use float16 weights.

On a p3.2xlarge instance, this version completes about 23.9s/epoch. The PyTorch
version reports completing 24 epochs in 72s, which comes out to 3s/epoch. So
JAX version is currently about an order of magnitude slower.

What am I missing?

Differences with the original:
* We don't use any batchnorm for now. If anything that should make the JAX version faster.
* We use a slightly different optimizer.
* Data augmentation has been removed, as @levskaya suggested that might be slowing things down.
"""
import time
from contextlib import contextmanager

import augmax
import jax.nn
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, lax, random, value_and_grad, vmap

### Various utility functions
@contextmanager
def timeblock(name):
  start = time.time()
  try:
    yield
  finally:
    end = time.time()
    print(f"... {name} took {end - start:.5f} seconds")

class RngPooper:
  """A stateful wrapper around stateless random.PRNGKey's."""

  def __init__(self, init_rng):
    self.rng = init_rng

  def poop(self):
    self.rng, rng_key = random.split(self.rng)
    return rng_key

### Model definition
dtype = jnp.float16

# See https://github.com/samuela/cifar10-fast/blob/master/dawn_utils.py.
class ResNetModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    # prep
    print("init            ", x.shape)
    x = nn.Conv(features=64, kernel_size=(3, 3), dtype=dtype)(x)
    # batchnorm
    print("after Conv_0    ", x.shape)
    x = nn.relu(x)
    # no pool in the "prep" layer

    # layer1
    x = nn.Conv(features=128, kernel_size=(3, 3), dtype=dtype)(x)
    print("after Conv_1    ", x.shape)
    # batchnorm
    x = nn.relu(x)
    x = nn.max_pool(x, (2, 2), strides=(2, 2))
    print("after max_pool_0", x.shape)
    residual = x
    y = nn.Conv(features=128, kernel_size=(3, 3), dtype=dtype)(x)
    print("after Conv_2    ", y.shape)
    # batchnorm
    y = nn.relu(y)
    y = nn.Conv(features=128, kernel_size=(3, 3), dtype=dtype)(y)
    print("after Conv_3    ", y.shape)
    # batchnorm
    y = nn.relu(y)
    x = y + residual

    # layer2
    x = nn.Conv(features=256, kernel_size=(3, 3), dtype=dtype)(x)
    print("after Conv_4    ", x.shape)
    # batchnorm
    x = nn.relu(x)
    x = nn.max_pool(x, (2, 2), strides=(2, 2))
    print("after max_pool_1", x.shape)

    # layer3
    x = nn.Conv(features=512, kernel_size=(3, 3), dtype=dtype)(x)
    print("after Conv_5    ", x.shape)
    # batchnorm
    x = nn.relu(x)
    x = nn.max_pool(x, (2, 2), strides=(2, 2))
    print("after max_pool_2", x.shape)
    residual = x
    y = nn.Conv(features=512, kernel_size=(3, 3), dtype=dtype)(x)
    print("after Conv_6    ", y.shape)
    # batchnorm
    y = nn.relu(y)
    y = nn.Conv(features=512, kernel_size=(3, 3), dtype=dtype)(y)
    print("after Conv_7    ", y.shape)
    # batchnorm
    y = nn.relu(y)
    x = y + residual

    x = nn.max_pool(x, (4, 4), strides=(4, 4))
    print("after max_pool_3", x.shape)
    x = jnp.reshape(x, (x.shape[0], -1))
    x = nn.Dense(10, dtype=dtype, use_bias=False)(x)
    x = nn.log_softmax(x)
    return x

### Train loop, etc
def make_stuff(model, train_ds, batch_size: int):
  ds_images, ds_labels = train_ds
  # `lax.scan` requires that all the batches have identical shape so we have to
  # skip the final batch if it is incomplete.
  num_train_examples = ds_labels.shape[0]
  assert num_train_examples >= batch_size
  num_batches = num_train_examples // batch_size

  # Applied to all input images, test and train.
  normalize_transform = augmax.Chain(augmax.ByteToFloat(), augmax.Normalize())

  @jit
  def batch_eval(params, images, labels):
    images_f32 = vmap(normalize_transform)(None, images)
    y_onehot = jax.nn.one_hot(labels, 10)
    logits = model.apply({"params": params}, images_f32)
    l = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_onehot))
    num_correct = jnp.sum(jnp.argmax(logits, axis=-1) == labels)
    return l, num_correct

  @jit
  def train_epoch(rng, train_state):
    batch_ix = random.permutation(rng, num_train_examples)[:num_batches * batch_size].reshape(
        (num_batches, batch_size))

    def step(train_state, i):
      p = batch_ix[i, :]
      images = ds_images[p, :, :, :]
      # images = jnp.zeros((batch_size, 32, 32, 3), dtype=jnp.uint8)
      labels = ds_labels[p]
      (l, num_correct), g = value_and_grad(batch_eval, has_aux=True)(train_state.params, images,
                                                                     labels)
      return train_state.apply_gradients(grads=g), (l, num_correct)

    # `lax.scan` is tricky to use correctly. See https://github.com/google/jax/discussions/9669#discussioncomment-2234793.
    train_state, (losses, num_corrects) = lax.scan(step, train_state, jnp.arange(num_batches))
    return train_state, (jnp.mean(batch_size * losses),
                         jnp.sum(num_corrects) / (num_batches * batch_size))

  def dataset_loss_and_accuracy(params, dataset, batch_size: int):
    images, labels = dataset
    num_examples = images.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    batch_ix = jnp.arange(num_examples).reshape((num_batches, batch_size))
    # Can't use vmap or run in a single batch since that overloads GPU memory.
    losses, num_corrects = zip(*[
        batch_eval(params, images[batch_ix[i, :], :, :, :], labels[batch_ix[i, :]])
        for i in range(num_batches)
    ])
    losses = jnp.array(losses)
    num_corrects = jnp.array(num_corrects)
    return jnp.mean(batch_size * losses), jnp.sum(num_corrects) / num_examples

  ret = lambda: None
  ret.batch_eval = batch_eval
  ret.train_epoch = train_epoch
  ret.dataset_loss_and_accuracy = dataset_loss_and_accuracy
  return ret

def get_datasets():
  """Return the training and test datasets, as jnp.array's."""
  import tensorflow as tf

  # See https://github.com/google/jax/issues/9454.
  tf.config.set_visible_devices([], "GPU")
  import tensorflow_datasets as tfds

  train_ds = tfds.load("cifar10", split="train", as_supervised=True)
  test_ds = tfds.load("cifar10", split="test", as_supervised=True)

  train_ds = tfds.as_numpy(train_ds)
  test_ds = tfds.as_numpy(test_ds)

  train_images = jnp.stack([x for x, _ in train_ds])
  train_labels = jnp.stack([y for _, y in train_ds])
  test_images = jnp.stack([x for x, _ in test_ds])
  test_labels = jnp.stack([y for _, y in test_ds])

  return (train_images, train_labels), (test_images, test_labels)

def init_train_state(rng, learning_rate, model):
  tx = optax.sgd(learning_rate, momentum=0.9)
  vars = model.init(rng, jnp.zeros((1, 32, 32, 3)))
  return TrainState.create(apply_fn=model.apply, params=vars["params"], tx=tx)

if __name__ == "__main__":
  batch_size = 512

  rp = RngPooper(random.PRNGKey(123))

  model = ResNetModel()
  train_ds, test_ds = get_datasets()
  stuff = make_stuff(model, train_ds, batch_size)
  train_state = init_train_state(rp.poop(), learning_rate=0.001, model=model)

  print("Burn-in...")
  for epoch in range(5):
    with timeblock(f"Burn-in epoch"):
      train_state, (train_loss, train_accuracy) = stuff.train_epoch(rp.poop(), train_state)
      test_loss, test_accuracy = stuff.dataset_loss_and_accuracy(train_state.params,
                                                                 test_ds,
                                                                 batch_size=1000)

  print("Training...")
  for epoch in range(10):
    with timeblock(f"Train epoch"):
      with jax.profiler.trace(log_dir="./logs"):
        train_state, (train_loss, train_accuracy) = stuff.train_epoch(rp.poop(), train_state)
        train_loss.block_until_ready()
        train_accuracy.block_until_ready()
    with timeblock("Test eval"):
      test_loss, test_accuracy = stuff.dataset_loss_and_accuracy(train_state.params,
                                                                 test_ds,
                                                                 batch_size=1000)
    print(
        f"Epoch {epoch}: train loss {train_loss:.3f}, train accuracy {train_accuracy:.3f}, test loss {test_loss:.3f}, test accuracy {test_accuracy:.3f}"
    )
