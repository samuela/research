"""TODO:
* implement "gains" module
* implement L1 prox operator
"""
import itertools
import time
from typing import NamedTuple

import flax
import jax.numpy as jnp
import numpy.random as npr
import optax
from flax import linen as nn
from flax import traverse_util
from flax.core import freeze, unfreeze
from jax import grad, jit, random, value_and_grad
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, LogSoftmax, Relu, Tanh
from jax.flatten_util import ravel_pytree
from jax.lax import stop_gradient
from jax.nn.initializers import glorot_normal, normal
from jax.tree_util import tree_flatten, tree_map, tree_unflatten
from tqdm import tqdm

import jax_examples_datasets as datasets
import wandb
from utils import RngPooper, l1prox

def partition(pred, iterable):
  trues = []
  falses = []
  for item in iterable:
    if pred(item):
      trues.append(item)
    else:
      falses.append(item)
  return trues, falses

def partition_dict(pred, d):
  trues = {}
  falses = {}
  for k, v in d.items():
    if pred(k):
      trues[k] = v
    else:
      falses[k] = v
  return trues, falses

def merge_params(a, b):
  combined_params = {**a, **b}
  return freeze(
      traverse_util.unflatten_dict({tuple(k.split("/")): v
                                    for k, v in combined_params.items()}))

class _net(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Dense(1024)(x)
    x = nn.relu(x)
    x = nn.Dense(1024)(x)
    x = nn.relu(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x

net = _net()

if __name__ == "__main__":
  wandb.init(project="playing-the-lottery", entity="skainswo")

  config = wandb.config
  config.learning_rate = 0.001
  config.num_epochs = 100
  config.batch_size = 128
  # config.momentum_mass = 0.9
  # config.remove_percentile = 80
  # config.l1_lambda = 10000

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

  def train(rng, trainable_predicate, log_prefix):
    def loss(trainable_params, untrainable_params, batch):
      inputs, targets = batch
      preds = net.apply(merge_params(trainable_params, untrainable_params), inputs)
      return -jnp.mean(jnp.sum(preds * targets, axis=1))

    def accuracy(trainable_params, untrainable_params, batch):
      inputs, targets = batch
      target_class = jnp.argmax(targets, axis=1)
      params = merge_params(trainable_params, untrainable_params)
      predicted_class = jnp.argmax(net.apply(params, inputs), axis=1)
      return jnp.mean(predicted_class == target_class)

    tx = optax.adam(config.learning_rate)

    @jit
    def update(opt_state, trainable_params, untrainable_params, batch):
      batch_loss, g = value_and_grad(loss)(trainable_params, untrainable_params, batch)
      # Standard gradient update on the smooth part.
      updates, opt_state = tx.update(g, opt_state)
      trainable_params = optax.apply_updates(trainable_params, updates)
      # Proximal update on the L1 non-smooth part.
      # TODO
      return opt_state, trainable_params, untrainable_params, batch_loss

    params = net.init(rng, jnp.zeros((1, 28 * 28)))
    flat_params = {"/".join(k): v for k, v in traverse_util.flatten_dict(unfreeze(params)).items()}
    trainable_params, untrainable_params = partition_dict(trainable_predicate, flat_params)
    print(tree_map(jnp.shape, trainable_params))
    opt_state = tx.init(trainable_params)
    itercount = itertools.count()
    batches = data_stream()
    start_time = time.time()
    for epoch in tqdm(range(config.num_epochs)):
      for _ in range(num_batches):
        step = next(itercount)
        opt_state, trainable_params, untrainable_params, batch_loss = update(
            opt_state, trainable_params, untrainable_params, next(batches))
        wandb.log({
            f"{log_prefix}/batch_loss": batch_loss,
            "step": step,
            "wallclock": time.time() - start_time
        })

      # Calculate the proportion of gains that are dead.
      # gains, _ = ravel_pytree(
      #     tree_map(lambda x: x.gain if isinstance(x, ProximalGainLayerWeights) else jnp.array([]),
      #              params,
      #              is_leaf=lambda x: isinstance(x, ProximalGainLayerWeights)))
      # dead_units_proportion = jnp.sum(jnp.abs(gains) < 1e-12) / jnp.size(gains)
      # print(dead_units_proportion)

      wandb.log({
          f"{log_prefix}/train_loss":
          loss(trainable_params, untrainable_params, (train_images, train_labels)),
          f"{log_prefix}/test_loss":
          loss(trainable_params, untrainable_params, (test_images, test_labels)),
          f"{log_prefix}/train_accuracy":
          accuracy(trainable_params, untrainable_params, (train_images, train_labels)),
          f"{log_prefix}/test_accuracy":
          accuracy(trainable_params, untrainable_params, (test_images, test_labels)),
          # f"{log_prefix}/dead_units_proportion": dead_units_proportion,
          "step":
          step,
          "epoch":
          epoch,
          "wallclock":
          time.time() - start_time
      })

    return params

  print("Training normal model...")
  train(random.PRNGKey(0), trainable_predicate=lambda k: True, log_prefix="normal")

  print("Training only weights model...")
  train(random.PRNGKey(0),
        trainable_predicate=lambda k: k.endswith("/kernel"),
        log_prefix="only_weights")

  print("Training only biases model...")
  train(random.PRNGKey(0),
        trainable_predicate=lambda k: k.endswith("/bias"),
        log_prefix="only_biases")

  # print("Training L1 gain with dense_gradients model...")
  # final_params = train(random.PRNGKey(0),
  #                      l1_lambda=config.l1_lambda,
  #                      dense_gradients=True,
  #                      log_prefix="l1_gain_dense")

  # print("Training L1 gain without dense_gradients model...")
  # final_params = train(random.PRNGKey(0),
  #                      l1_lambda=config.l1_lambda,
  #                      dense_gradients=False,
  #                      log_prefix="l1_gain")
