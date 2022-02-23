"""Train a convnet on CIFAR-10 on one random seed. Serialize the model for
interpolation downstream.

Notes:
* This convnet is something of my own creation
* flax example code used to have a CIFAR-10 example but it seems to have gone missing: https://github.com/google/flax/issues/122#issuecomment-1032108906
* Example VGG/CIFAR-10 model in flax: https://github.com/rolandgvc/flaxvision/blob/master/flaxvision/models/vgg.py
* A good reference in PyTorch is https://github.com/kuangliu/pytorch-cifar
"""
import argparse

import augmax
import jax.nn
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.checkpoints import restore_checkpoint, save_checkpoint
from flax.training.train_state import TrainState
from jax import jit, lax, random, value_and_grad, vmap
from tqdm import tqdm

import wandb
from utils import RngPooper, ec2_get_instance_type, timeblock

# See https://github.com/tensorflow/tensorflow/issues/53831.

activation = nn.relu

class TestModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=8, kernel_size=(3, 3))(x)
    x = activation(x)
    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = activation(x)
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = activation(x)
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = activation(x)

    x = jnp.mean(x, axis=-1)
    x = jnp.reshape(x, (x.shape[0], -1))
    x = nn.Dense(32)(x)
    x = activation(x)
    x = nn.Dense(32)(x)
    x = activation(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x

class ConvNetModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = activation(x)
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = activation(x)
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = activation(x)
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = activation(x)
    # Take the mean along the channel dimension. Otherwise the following dense
    # layer is massive.
    x = jnp.mean(x, axis=-1)
    x = jnp.reshape(x, (x.shape[0], -1))
    x = nn.Dense(4096)(x)
    x = activation(x)
    x = nn.Dense(4096)(x)
    x = activation(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x

def make_batcher(num_examples: int, batch_size: int):
  # We need to special case the situation where batch_size divides num_examples,
  # since in that situation `jnp.split` will return an empty array as the final
  # batch.
  if num_examples % batch_size != 0:
    splits = list(jnp.arange(1, num_examples // batch_size + 1) * batch_size)
  else:
    splits = list(jnp.arange(1, num_examples // batch_size) * batch_size)

  return lambda arr: jnp.split(arr, splits)

def test_make_batcher():
  # there's an odd-shaped batch at the end
  for fn in [make_batcher(5, 2), jit(make_batcher(5, 2))]:
    assert len(fn(jnp.array([1, 2, 3, 4, 5]))) == 3
    assert len(fn(jnp.array([1, 2, 3, 4, 5]))[0]) == 2
    assert len(fn(jnp.array([1, 2, 3, 4, 5]))[1]) == 2
    assert len(fn(jnp.array([1, 2, 3, 4, 5]))[2]) == 1

  # no odd-shaped batch at the end
  for fn in [make_batcher(4, 2), jit(make_batcher(4, 2))]:
    assert len(fn(jnp.array([1, 2, 3, 4]))) == 2
    assert len(fn(jnp.array([1, 2, 3, 4]))[0]) == 2
    assert len(fn(jnp.array([1, 2, 3, 4]))[1]) == 2

  # batch_size == num_examples
  for fn in [make_batcher(2, 2), jit(make_batcher(2, 2))]:
    assert len(fn(jnp.array([1, 2]))) == 1
    assert len(fn(jnp.array([1, 2]))[0]) == 2

  # batch_size > num_examples
  for fn in [make_batcher(2, 3), jit(make_batcher(2, 3))]:
    assert len(fn(jnp.array([1, 2]))) == 1
    assert len(fn(jnp.array([1, 2]))[0]) == 2

def make_batcher_in_paradise(num_examples: int, batch_size: int):
  """Like `make_batcher`, but skips the final batch if it is incomplete."""
  assert num_examples >= batch_size
  num_batches = num_examples // batch_size
  return lambda arr: jnp.split(arr[:num_batches * batch_size], num_batches)

def test_make_batcher_in_paradise():
  # there's an odd-shaped batch at the end
  for fn in [make_batcher_in_paradise(5, 2), jit(make_batcher_in_paradise(5, 2))]:
    assert len(fn(jnp.array([1, 2, 3, 4, 5]))) == 2
    assert len(fn(jnp.array([1, 2, 3, 4, 5]))[0]) == 2
    assert len(fn(jnp.array([1, 2, 3, 4, 5]))[1]) == 2

  # no odd-shaped batch at the end
  for fn in [make_batcher_in_paradise(4, 2), jit(make_batcher_in_paradise(4, 2))]:
    assert len(fn(jnp.array([1, 2, 3, 4]))) == 2
    assert len(fn(jnp.array([1, 2, 3, 4]))[0]) == 2
    assert len(fn(jnp.array([1, 2, 3, 4]))[1]) == 2

  # batch_size == num_examples
  for fn in [make_batcher_in_paradise(2, 2), jit(make_batcher_in_paradise(2, 2))]:
    assert len(fn(jnp.array([1, 2]))) == 1
    assert len(fn(jnp.array([1, 2]))[0]) == 2

def make_stuff(model, train_ds, batch_size: int):
  ds_images, ds_labels = train_ds
  num_train_examples = ds_labels.shape[0]
  # `lax.scan` requires that all the batches have identical shape so we have to
  # skip the final batch if it is incomplete.
  batcher = make_batcher_in_paradise(num_train_examples, batch_size)

  # Note: Some confusion regarding exactly what RandomCrop does: https://github.com/khdlr/augmax/issues/6
  train_transform = augmax.Chain(
      augmax.RandomCrop(32, 32),
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
    perm = random.permutation(rng1, num_train_examples)
    # We need rngs for data augmentation of each example.
    augmax_rngs = random.split(rng2, num_train_examples)

    def step(train_state, p):
      # See https://github.com/google/jax/issues/4564 as to why the array conversion is necessary.
      p = jnp.array(p)
      images = ds_images[p, :, :, :]
      labels = ds_labels[p]
      images_transformed = vmap(train_transform)(augmax_rngs[p], images)
      (l, num_correct), g = value_and_grad(batch_eval, has_aux=True)(train_state.params,
                                                                     images_transformed, labels)
      return train_state.apply_gradients(grads=g), (l, num_correct)

    train_state, (losses, num_corrects) = lax.scan(step, train_state, batcher(perm))
    return train_state, (jnp.mean(batch_size * losses), jnp.sum(num_corrects) / num_train_examples)

  def dataset_loss_and_accuracy(params, dataset, batch_size: int):
    images, labels = dataset
    num_examples = images.shape[0]
    batch = make_batcher(num_examples, batch_size)
    # Can't use vmap or run in a single batch since that overloads GPU memory.
    losses, num_corrects = zip(
        *[batch_eval(params, x, y) for x, y in zip(batch(images), batch(labels))])
    losses = jnp.array(losses)
    num_corrects = jnp.array(num_corrects)
    assert num_examples % batch_size == 0
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
  steps_per_epoch = jnp.ceil(num_train_examples / batch_size)
  lr_schedule = optax.cosine_decay_schedule(learning_rate, decay_steps=num_epochs * steps_per_epoch)
  tx = optax.sgd(lr_schedule, momentum=0.9)
  vars = model.init(rng, jnp.zeros((1, 32, 32, 3)))
  return TrainState.create(apply_fn=model.apply, params=vars["params"], tx=tx)

if __name__ == "__main__":
  with timeblock("tests"):
    test_make_batcher()
    test_make_batcher_in_paradise()

  parser = argparse.ArgumentParser()
  parser.add_argument("--test", action="store_true", help="Run in smoke-test mode")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--resume", type=str, help="wandb run to resume from (eg. 1kqqa9js)")
  parser.add_argument("--resume-epoch",
                      type=int,
                      help="The epoch to resume from. Required if --resume is set.")
  args = parser.parse_args()

  wandb.init(project="playing-the-lottery",
             entity="skainswo",
             tags=["cifar10", "convnet"],
             resume="must" if args.resume is not None else None,
             id=args.resume,
             mode="disabled" if args.test or False else "online")

  # Note: hopefully it's ok that we repeat this even when resuming a run?
  config = wandb.config
  config.ec2_instance_type = ec2_get_instance_type()
  config.test = args.test
  config.seed = args.seed
  config.learning_rate = 0.001
  config.num_epochs = 3 if config.test else 200
  config.batch_size = 7 if config.test else 512

  rp = RngPooper(random.PRNGKey(config.seed))

  model = TestModel() if config.test else ConvNetModel()
  train_ds, test_ds = get_datasets(config.test)
  stuff = make_stuff(model, train_ds, config.batch_size)
  train_state = init_train_state(rp.poop(),
                                 learning_rate=config.learning_rate,
                                 model=model,
                                 num_epochs=config.num_epochs,
                                 batch_size=config.batch_size,
                                 num_train_examples=train_ds[0].shape[0])
  start_epoch = 0

  if args.resume is not None:
    # Bring the the desired resume epoch into the wandb run directory so that it
    # can then be picked up by `restore_checkpoint` below.
    wandb.restore(f"checkpoint_{args.resume_epoch}")
    last_epoch, train_state = restore_checkpoint(wandb.run.dir, (0, train_state))
    # We need to increment last_epoch, because we store `(i, train_state)`
    # where `train_state` is the state _after_ i'th epoch. So we're actually
    # starting from the next epoch.
    start_epoch = last_epoch + 1

  for epoch in tqdm(range(start_epoch, config.num_epochs),
                    initial=start_epoch,
                    total=config.num_epochs):
    with timeblock(f"Train epoch"):
      train_state, (train_loss, train_accuracy) = stuff.train_epoch(rp.poop(), train_state)
    with timeblock("Test eval"):
      test_loss, test_accuracy = stuff.dataset_loss_and_accuracy(
          train_state.params, test_ds, batch_size=10 if config.test else 1000)

    if not config.test:
      with timeblock("Save checkpoint"):
        # See https://docs.wandb.ai/guides/track/advanced/save-restore
        save_checkpoint(wandb.run.dir, (epoch, train_state), epoch, keep_every_n_steps=10)

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    })
