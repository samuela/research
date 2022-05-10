"""Train an MLP on MNIST on one random seed. Serialize the model for
interpolation downstream."""
import argparse

import augmax
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training.checkpoints import restore_checkpoint, save_checkpoint
from flax.training.train_state import TrainState
from jax import jit, random, value_and_grad, vmap
from tqdm import tqdm

import wandb
from utils import RngPooper, ec2_get_instance_type, timeblock

# See https://github.com/tensorflow/tensorflow/issues/53831.

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

activation = nn.relu

class TestModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = jnp.reshape(x, (-1, 28 * 28))
    x = nn.Dense(1024)(x)
    x = activation(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x

class MLPModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = jnp.reshape(x, (-1, 28 * 28))
    x = nn.Dense(512)(x)
    x = activation(x)
    x = nn.Dense(512)(x)
    x = activation(x)
    x = nn.Dense(512)(x)
    x = activation(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x

def make_stuff(model):
  normalize_transform = augmax.ByteToFloat()

  @jit
  def batch_eval(params, images_u8, labels):
    images_f32 = vmap(normalize_transform)(None, images_u8)
    logits = model.apply({"params": params}, images_f32)
    y_onehot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y_onehot))
    num_correct = jnp.sum(jnp.argmax(logits, axis=-1) == jnp.argmax(y_onehot, axis=-1))
    return loss, num_correct

  @jit
  def step(train_state, images_f32, labels):
    (l, num_correct), g = value_and_grad(batch_eval, has_aux=True)(train_state.params, images_f32,
                                                                   labels)
    return train_state.apply_gradients(grads=g), l

  def dataset_loss_and_num_correct_jnp(params, dataset, batch_size: int):
    images_u8, labels = dataset
    num_examples = images_u8.shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    batch_ix = jnp.arange(num_examples).reshape((num_batches, batch_size))
    # Can't use vmap or run in a single batch since that overloads GPU memory.
    losses, num_corrects = zip(*[
        batch_eval(params, images_u8[batch_ix[i, :], :, :, :], labels[batch_ix[i, :]])
        for i in range(num_batches)
    ])
    losses = jnp.array(losses)
    num_corrects = jnp.array(num_corrects)
    return jnp.sum(batch_size * losses) / num_examples, jnp.sum(num_corrects) / num_examples

  def dataset_loss_and_num_correct_tfds(params, dataset):
    dataset_total_size = sum([x.shape[0] for x, _ in dataset])
    return (jnp.sum(jnp.array([x.shape[0] * batch_eval(params, x, y)[0]
                               for x, y in dataset])) / dataset_total_size,
            jnp.sum(jnp.array([batch_eval(params, x, y)[1]
                               for x, y in dataset])) / dataset_total_size)

  ret = lambda: None
  ret.batch_eval = batch_eval
  ret.step = step
  ret.dataset_loss_and_num_correct_jnp = dataset_loss_and_num_correct_jnp
  ret.dataset_loss_and_num_correct_tfds = dataset_loss_and_num_correct_tfds
  return ret

def get_datasets_tfds(smoke_test_mode):
  """Return the training and test datasets, unbatched.

  smoke_test_mode: Whether or not we're running in "smoke test" mode.
  """
  train_ds = tfds.load("mnist", split="train", as_supervised=True)
  test_ds = tfds.load("mnist", split="test", as_supervised=True)
  # Note: The take/cache warning:
  #     2022-01-25 07:32:58.144059: W tensorflow/core/kernels/data/cache_dataset_ops.cc:768] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
  # is not because we're actually doing this in the wrong order, but rather that
  # the dataset is loaded in and called .cache() on before we receive it.
  if smoke_test_mode:
    train_ds = train_ds.take(13)
    test_ds = test_ds.take(17)

  return train_ds, test_ds

def get_datasets_jnp(smoke_test_mode):
  train_ds, test_ds = get_datasets_tfds(smoke_test_mode)

  train_ds = tfds.as_numpy(train_ds)
  test_ds = tfds.as_numpy(test_ds)

  train_images = jnp.stack([x for x, _ in train_ds])
  train_labels = jnp.stack([y for _, y in train_ds])
  test_images = jnp.stack([x for x, _ in test_ds])
  test_labels = jnp.stack([y for _, y in test_ds])

  return (train_images, train_labels), (test_images, test_labels)

def init_train_state(rng, learning_rate, model):
  tx = optax.adam(learning_rate)
  vars = model.init(rng, jnp.zeros((1, 28, 28, 1)))
  return TrainState.create(apply_fn=model.apply, params=vars["params"], tx=tx)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", action="store_true", help="Run in smoke-test mode")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  args = parser.parse_args()

  wandb.init(project="playing-the-lottery",
             entity="skainswo",
             tags=["mnist", "mlp"],
             mode="disabled" if args.test else "online")

  # Note: hopefully it's ok that we repeat this even when resuming a run?
  config = wandb.config
  config.ec2_instance_type = ec2_get_instance_type()
  config.test = args.test
  config.seed = args.seed
  config.learning_rate = 0.001
  config.num_epochs = 10 if config.test else 50
  config.batch_size = 7 if config.test else 500

  rp = RngPooper(random.PRNGKey(config.seed))

  model = TestModel() if config.test else MLPModel()
  stuff = make_stuff(model)

  train_ds_tfds, test_ds_tfds = get_datasets_tfds(smoke_test_mode=config.test)
  train_ds_jnp, test_ds_jnp = get_datasets_jnp(smoke_test_mode=config.test)
  num_train_examples = train_ds_jnp[0].shape[0]
  num_test_examples = test_ds_jnp[0].shape[0]

  assert num_train_examples % config.batch_size == 0

  train_state = init_train_state(rp.poop(), config.learning_rate, model)

  for epoch in tqdm(range(config.num_epochs)):
    with timeblock(f"Epoch"):
      batch_ix = random.permutation(rp.poop(), num_train_examples).reshape((-1, config.batch_size))
      for i in range(batch_ix.shape[0]):
        p = batch_ix[i, :]
        images_u8 = train_ds_jnp[0][p, :, :, :]
        labels = train_ds_jnp[1][p]
        train_state, batch_loss = stuff.step(train_state, images_u8, labels)

    # Evaluate train/test loss/accuracy
    with timeblock("Model eval (jnp)"):
      train_loss_jnp, train_accuracy_jnp = stuff.dataset_loss_and_num_correct_jnp(
          train_state.params, train_ds_jnp, 1000)
      test_loss_jnp, test_accuracy_jnp = stuff.dataset_loss_and_num_correct_jnp(
          train_state.params, test_ds_jnp, 1000)

    with timeblock("Model eval (tfds)"):
      train_ds_tfds_batched = tfds.as_numpy(train_ds_tfds.batch(1000))
      test_ds_tfds_batched = tfds.as_numpy(test_ds_tfds.batch(1000))

      train_loss_tfds, train_accuracy_tfds = stuff.dataset_loss_and_num_correct_tfds(
          train_state.params, train_ds_tfds_batched)
      test_loss_tfds, test_accuracy_tfds = stuff.dataset_loss_and_num_correct_tfds(
          train_state.params, test_ds_tfds_batched)

    wandb.log({
        "epoch": epoch,
        "train_loss_jnp": train_loss_jnp,
        "test_loss_jnp": test_loss_jnp,
        "train_accuracy_jnp": train_accuracy_jnp,
        "test_accuracy_jnp": test_accuracy_jnp,
        "train_loss_tfds": train_loss_tfds,
        "test_loss_tfds": test_loss_tfds,
        "train_accuracy_tfds": train_accuracy_tfds,
        "test_accuracy_tfds": test_accuracy_tfds,
    })
