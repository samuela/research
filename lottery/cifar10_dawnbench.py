"""Replicate https://github.com/davidcpage/cifar10-fast in JAX.

Some notes:
* Entirety of the CIFAR-10 dataset is loaded into GPU memory, for speeeedz.
* Training epochs are fully jit'd with `lax.scan`.
* Data augmentation is fully jit'd thanks to augmax.
* Like the pytorch version, we use float16 weights.

On a p3.2xlarge instance, this version completes about 23.9s/epoch. The PyTorch
version reports completing 24 epochs in 72s, which comes out to 3s/epoch. So
JAX version is currently about an order of magnitude slower.

What am I missing?

Differences with the original:
* We don't use any batchnorm for now. If anything that should make the JAX version faster.
* We use a slightly different optimizer.
"""
import argparse
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

class ResNetModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    # prep
    x = nn.Conv(features=64, kernel_size=(3, 3), dtype=dtype)(x)
    x = nn.relu(x)

    # layer1
    x = nn.Conv(features=128, kernel_size=(3, 3), dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (2, 2))
    residual = x
    y = nn.Conv(features=128, kernel_size=(3, 3), dtype=dtype)(x)
    y = nn.relu(y)
    y = nn.Conv(features=128, kernel_size=(3, 3), dtype=dtype)(y)
    y = nn.relu(y)
    x = y + residual

    # layer2
    x = nn.Conv(features=256, kernel_size=(3, 3), dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (2, 2))

    # layer3
    x = nn.Conv(features=512, kernel_size=(3, 3), dtype=dtype)(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (2, 2))
    residual = x
    y = nn.Conv(features=512, kernel_size=(3, 3), dtype=dtype)(x)
    y = nn.relu(y)
    y = nn.Conv(features=512, kernel_size=(3, 3), dtype=dtype)(y)
    y = nn.relu(y)
    x = y + residual

    x = nn.max_pool(x, (4, 4))
    x = jnp.reshape(x, (x.shape[0], -1))
    x = nn.Dense(10, dtype=dtype)(x)
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

  train_transform = augmax.Chain(
      # augmax does not seem to support random crops with padding. See https://github.com/khdlr/augmax/issues/6.
      # augmax.RandomCrop(32, 32),
      augmax.HorizontalFlip(),
      augmax.Rotate(),
  )
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
    rng1, rng2 = random.split(rng)
    batch_ix = random.permutation(rng1, num_train_examples)[:num_batches * batch_size].reshape(
        (num_batches, batch_size))
    # We need rngs for data augmentation of each example.
    augmax_rngs = random.split(rng2, num_batches * batch_size)

    def step(train_state, i):
      p = batch_ix[i, :]
      images = ds_images[p, :, :, :]
      labels = ds_labels[p]
      images_transformed = vmap(train_transform)(augmax_rngs[p], images)
      (l, num_correct), g = value_and_grad(batch_eval, has_aux=True)(train_state.params,
                                                                     images_transformed, labels)
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

def get_datasets(test: bool):
  """Return the training and test datasets, as jnp.array's."""
  if test:
    num_train = 100
    num_test = 1000
    train_images = random.choice(random.PRNGKey(0), jnp.arange(256, dtype=jnp.uint8),
                                 (num_train, 32, 32, 3))
    test_images = random.choice(random.PRNGKey(1), jnp.arange(256, dtype=jnp.uint8),
                                (num_test, 32, 32, 3))
    train_labels = random.choice(random.PRNGKey(2), jnp.arange(10, dtype=jnp.uint8), (num_train, ))
    test_labels = random.choice(random.PRNGKey(3), jnp.arange(10, dtype=jnp.uint8), (num_test, ))
    return (train_images, train_labels), (test_images, test_labels)
  else:
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

def init_train_state(rng, learning_rate, model, num_epochs, batch_size, num_train_examples):
  # See https://github.com/kuangliu/pytorch-cifar.
  steps_per_epoch = num_train_examples // batch_size
  lr_schedule = optax.cosine_decay_schedule(learning_rate, decay_steps=num_epochs * steps_per_epoch)
  tx = optax.sgd(lr_schedule, momentum=0.9)
  vars = model.init(rng, jnp.zeros((1, 32, 32, 3)))
  return TrainState.create(apply_fn=model.apply, params=vars["params"], tx=tx)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  args = parser.parse_args()

  # Note: hopefully it's ok that we repeat this even when resuming a run?
  config = lambda: None
  config.test = False
  config.seed = args.seed
  config.learning_rate = 0.001
  config.num_epochs = 10
  config.batch_size = 10 if config.test else 512

  rp = RngPooper(random.PRNGKey(config.seed))

  model = ResNetModel()
  train_ds, test_ds = get_datasets(test=config.test)
  stuff = make_stuff(model, train_ds, config.batch_size)
  train_state = init_train_state(rp.poop(),
                                 learning_rate=config.learning_rate,
                                 model=model,
                                 num_epochs=config.num_epochs,
                                 batch_size=config.batch_size,
                                 num_train_examples=train_ds[0].shape[0])

  print("Burn-in...")
  for epoch in range(3):
    train_state, (train_loss, train_accuracy) = stuff.train_epoch(rp.poop(), train_state)
    test_loss, test_accuracy = stuff.dataset_loss_and_accuracy(train_state.params,
                                                               test_ds,
                                                               batch_size=1000)

  print("Training...")
  for epoch in range(config.num_epochs):
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
