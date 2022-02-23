"""Train a convnet on CIFAR-10 on one random seed. Serialize the model for
interpolation downstream.

Notes:
* flax example code used to have a CIFAR-10 example but it seems to have gone missing: https://github.com/google/flax/issues/122#issuecomment-1032108906
* Example VGG/CIFAR-10 model in flax: https://github.com/rolandgvc/flaxvision/blob/master/flaxvision/models/vgg.py
* A good reference in PyTorch is https://github.com/kuangliu/pytorch-cifar

Things to try:
* weight decay
* resnet18
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

class TestModel(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=8, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=16, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)

    x = jnp.mean(x, axis=-1)
    x = jnp.reshape(x, (x.shape[0], -1))
    x = nn.Dense(32)(x)
    x = nn.relu(x)
    x = nn.Dense(32)(x)
    x = nn.relu(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x

class ConvNetModel(nn.Module):
  """This convnet is something of my own creation. Doesn't seem to work all that
  well..."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.Conv(features=128, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    # Take the mean along the channel dimension. Otherwise the following dense
    # layer is massive.
    x = jnp.mean(x, axis=-1)
    x = jnp.reshape(x, (x.shape[0], -1))
    x = nn.Dense(4096)(x)
    x = nn.relu(x)
    x = nn.Dense(4096)(x)
    x = nn.relu(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x

class VGG16(nn.Module):

  @nn.compact
  def __call__(self, x):
    # Backbone
    for _ in range(2):
      x = nn.Conv(features=64, kernel_size=(3, 3))(x)
      x = nn.GroupNorm()(x)
      x = nn.relu(x)
    x = nn.max_pool(x, (2, 2), strides=(2, 2))
    for _ in range(2):
      x = nn.Conv(features=128, kernel_size=(3, 3))(x)
      x = nn.GroupNorm()(x)
      x = nn.relu(x)
    x = nn.max_pool(x, (2, 2), strides=(2, 2))
    for _ in range(3):
      x = nn.Conv(features=256, kernel_size=(3, 3))(x)
      x = nn.GroupNorm()(x)
      x = nn.relu(x)
    x = nn.max_pool(x, (2, 2), strides=(2, 2))
    for _ in range(3):
      x = nn.Conv(features=512, kernel_size=(3, 3))(x)
      x = nn.GroupNorm()(x)
      x = nn.relu(x)
    x = nn.max_pool(x, (2, 2), strides=(2, 2))
    for _ in range(3):
      x = nn.Conv(features=512, kernel_size=(3, 3))(x)
      x = nn.GroupNorm()(x)
      x = nn.relu(x)
    x = nn.max_pool(x, (2, 2), strides=(2, 2))

    # Classifier
    # Note: everyone seems to do a different thing here.
    # * https://github.com/davisyoshida/vgg16-haiku/blob/4ef0bd001bf9daa4cfb2fa83ea3956ec01add3a8/vgg/vgg.py#L56
    #     does average pooling with a kernel size of (7, 7)
    # * https://github.com/kuangliu/pytorch-cifar/blob/49b7aa97b0c12fe0d4054e670403a16b6b834ddd/models/vgg.py#L37
    #     does average pooling with a kernel size of (1, 1) which doesn't seem
    #     to accomplish anything. See https://github.com/kuangliu/pytorch-cifar/issues/110.
    #     But this paper also doesn't really do the dense layers the same as in
    #     the paper either...
    # * The paper itself doesn't mention any kind of pooling...
    #
    # I'll stick to replicating the paper as closely as possible for now.
    x = jnp.reshape(x, (x.shape[0], -1))
    x = nn.Dense(4096)(x)
    x = nn.Dense(4096)(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x

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
    # Note that the train accuracy calculation here is based on the number of
    # examples we've actually covered, not the number of train examples, since
    # we skip the last (ragged) batch.
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
  # TODO: add_decayed_weights
  vars = model.init(rng, jnp.zeros((1, 32, 32, 3)))
  return TrainState.create(apply_fn=model.apply, params=vars["params"], tx=tx)

if __name__ == "__main__":
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
             tags=["cifar10", "vgg16"],
             resume="must" if args.resume is not None else None,
             id=args.resume,
             mode="disabled" if args.test or False else "online")

  # Note: hopefully it's ok that we repeat this even when resuming a run?
  config = wandb.config
  config.ec2_instance_type = ec2_get_instance_type()
  config.test = args.test
  config.seed = args.seed
  config.learning_rate = 1e-3
  config.num_epochs = 3 if config.test else 200
  config.batch_size = 7 if config.test else 128

  rp = RngPooper(random.PRNGKey(config.seed))

  model = TestModel() if config.test else VGG16()
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

    # if not config.test:
    #   with timeblock("Save checkpoint"):
    #     # See https://docs.wandb.ai/guides/track/advanced/save-restore
    #     save_checkpoint(wandb.run.dir, (epoch, train_state), epoch, keep_every_n_steps=10)

    print(
        f"Epoch {epoch}: train loss {train_loss:.3f}, train accuracy {train_accuracy:.3f}, test loss {test_loss:.3f}, test accuracy {test_accuracy:.3f}"
    )
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    })
