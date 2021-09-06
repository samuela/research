"""Testing the original lottery ticket hypothesis.

Lottery ticket models degrade as the depth of the network increases, e.g. with 2
layers it works well, but with 8 layers it breaks down entirely. This holds even
though the data order is the same. Tanh activations seem to scale better with
depth than relu activations.

TODO:
  - implement prox update
"""
import itertools
import time

import jax.numpy as jnp
import numpy.random as npr
from jax import grad, jit, random, value_and_grad
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, LogSoftmax, Relu, Tanh
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from tqdm import tqdm

import jax_examples_datasets as datasets
import wandb
from utils import RngPooper

def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))

def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)

init_random_params, predict = stax.serial(
    Dense(1024),
    Tanh,
    Dense(1024),
    Tanh,
    # Dense(1024), Tanh,
    # Dense(1024), Tanh,
    # Dense(1024), Tanh,
    # Dense(1024), Tanh,
    # Dense(1024), Tanh,
    # Dense(1024), Tanh,
    Dense(10),
    LogSoftmax)

if __name__ == "__main__":
  wandb.init(project="playing-the-lottery", entity="skainswo")

  rp = RngPooper(random.PRNGKey(0))

  config = wandb.config
  config.learning_rate = 0.001
  config.num_epochs = 100
  config.batch_size = 128
  config.momentum_mass = 0.9
  config.remove_percentile = 80

  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, config.batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * config.batch_size:(i + 1) * config.batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  opt_init, opt_update, get_params = optimizers.momentum(config.learning_rate,
                                                         mass=config.momentum_mass)

  @jit
  def update(i, opt_state, batch, mask):
    params = get_params(opt_state)
    batch_loss, g = value_and_grad(loss)(params, batch)
    g_masked = tree_map(lambda a, b: a * b, g, mask)
    return opt_update(i, g_masked, opt_state), batch_loss

  _, init_params = init_random_params(rp.poop(), (-1, 28 * 28))

  def train(init_params, mask, log_prefix):
    opt_state = opt_init(tree_map(lambda a, b: a * b, init_params, mask))
    itercount = itertools.count()
    batches = data_stream()
    start_time = time.time()
    for epoch in tqdm(range(config.num_epochs)):
      for _ in range(num_batches):
        step = next(itercount)
        opt_state, batch_loss = update(step, opt_state, next(batches), mask)
        wandb.log({
            f"{log_prefix}/batch_loss": batch_loss,
            "step": step,
            "wallclock": time.time() - start_time
        })

      params = get_params(opt_state)
      train_loss = loss(params, (train_images, train_labels))
      test_loss = loss(params, (test_images, test_labels))
      train_acc = accuracy(params, (train_images, train_labels))
      test_acc = accuracy(params, (test_images, test_labels))
      # print("Epoch {}".format(epoch))
      # print("  Train loss            {}".format(train_loss))
      # print("  Test loss             {}".format(test_loss))
      # print("  Training set accuracy {}".format(train_acc))
      # print("  Test set accuracy     {}".format(test_acc))

      wandb.log({
          f"{log_prefix}/train_loss": train_loss,
          f"{log_prefix}/test_loss": test_loss,
          f"{log_prefix}/train_accuracy": train_acc,
          f"{log_prefix}/test_accuracy": test_acc,
          "step": step,
          "epoch": epoch,
          "wallclock": time.time() - start_time
      })

    return get_params(opt_state)

  print("Training normal model...")
  everything_mask = tree_map(lambda x: jnp.ones_like(x, dtype=jnp.dtype("bool")), init_params)
  final_params = train(init_params, everything_mask, "no_mask")

  # Mask as was implemented in the original paper
  print("Training lottery ticket model...")
  final_params_flat, unravel = ravel_pytree(final_params)
  cutoff = jnp.percentile(jnp.abs(final_params_flat), config.remove_percentile)
  mask = unravel(jnp.abs(final_params_flat) > cutoff)
  # See https://github.com/google/jax/issues/7809.
  mask = tree_map(lambda x: x > 0.5, mask)
  train(init_params, mask, "lottery_mask")

  # Totally random mask
  print("Training random mask model...")
  mask = unravel(
      random.uniform(rp.poop(), final_params_flat.shape) > config.remove_percentile / 100)
  mask = tree_map(lambda x: x > 0.5, mask)
  train(init_params, mask, "random_mask")
