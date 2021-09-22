"""TODO:
* implement L1 prox operator
"""
import itertools
import re
import time
from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy.random as npr
import optax
from flax import linen as nn
from flax import traverse_util
from flax.core import freeze, unfreeze
from jax import grad, jit, random, value_and_grad
from jax.flatten_util import ravel_pytree
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

def flatten_params(params):
  return {"/".join(k): v for k, v in traverse_util.flatten_dict(unfreeze(params)).items()}

def unflatten_params(flat_params):
  return freeze(
      traverse_util.unflatten_dict({tuple(k.split("/")): v
                                    for k, v in flat_params.items()}))

def merge_params(a, b):
  return unflatten_params({**a, **b})

def kmatch(pattern, key):
  regex = "^"
  i = 0
  while i < len(pattern):
    if pattern[i] == "*":
      if i + 1 < len(pattern) and pattern[i + 1] == "*":
        regex += "(.*)"
        i += 2
      else:
        regex += "([^\/]*)"
        i += 1
    else:
      regex += pattern[i]
      i += 1
  regex += "$"
  return re.fullmatch(regex, key)

assert kmatch("*", "a") is not None
assert kmatch("*", "a").group(0) == "a"
assert kmatch("*", "a").group(1) == "a"
assert kmatch("abc", "def") is None
assert kmatch("abc/*/ghi", "abc/def/ghi").group(1) == "def"
assert kmatch("abc/**/jkl", "abc/def/ghi/jkl").group(1) == "def/ghi"
assert kmatch("abc/*/jkl", "abc/def/ghi/jkl") is None
assert kmatch("**/*", "abc/def/ghi/jkl").group(1) == "abc/def/ghi"
assert kmatch("**/*", "abc/def/ghi/jkl").group(2) == "jkl"

class OGDense(nn.Module):
  features: int
  activation: Callable = nn.relu

  @nn.compact
  def __call__(self, x):
    gain = self.param("gain", lambda rng, shape: random.choice(rng, jnp.array([-1.0, 1.0]), shape),
                      (self.features, ))

    x = nn.Dense(self.features, name="Dense")(x)
    x = self.activation(x)
    x = gain * x
    return x

def make_net(layer_features):
  first, *rest = layer_features

  class _net(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = OGDense(first, name="first")(x)
      for n in rest:
        x = OGDense(n)(x)
      x = nn.Dense(10, name="last")(x)
      x = nn.log_softmax(x)
      return x

  return _net()

net = make_net([2048] * 6)

if __name__ == "__main__":
  wandb.init(project="playing-the-lottery", entity="skainswo")

  config = wandb.config
  config.learning_rate = 0.001
  config.num_epochs = 100
  config.batch_size = 128
  config.remove_percentile = 80
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

  def train(net, init_params, trainable_predicate, log_prefix):
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
      # TODO: Proximal update on the L1 non-smooth part.
      return opt_state, trainable_params, untrainable_params, batch_loss

    trainable_params, untrainable_params = partition_dict(trainable_predicate,
                                                          flatten_params(init_params))
    print("Trainable params:")
    print(tree_map(jnp.shape, trainable_params))
    assert len(trainable_params) > 0

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

    return merge_params(trainable_params, untrainable_params)

  # See https://github.com/google/jax/issues/7809.
  binarize = lambda arr: tree_map(lambda x: x > 0.5, arr)

  print("Training normal model...")
  net = make_net([2048] * 6)
  train(net,
        init_params=net.init(random.PRNGKey(0), jnp.zeros((1, 28 * 28))),
        trainable_predicate=lambda k: True,
        log_prefix="normal")

  print("Training only gains model...")
  net = make_net([2048] * 6)
  only_gains_final_params = train(net,
                                  init_params=net.init(random.PRNGKey(0), jnp.zeros((1, 28 * 28))),
                                  trainable_predicate=lambda k: kmatch("**/gain", k),
                                  log_prefix="only_gain")
  only_gains_final_params_flat = flatten_params(only_gains_final_params)
  print("  full model params:")
  print(tree_map(jnp.shape, only_gains_final_params_flat))
  gain_params = {k: v for k, v in only_gains_final_params_flat.items() if kmatch("**/gain", k)}
  gain_params_flat, unravel = ravel_pytree(gain_params)
  cutoff = jnp.percentile(jnp.abs(gain_params_flat), config.remove_percentile)
  gain_mask = binarize(unravel(jnp.abs(gain_params_flat) > cutoff))
  print(tree_map(jnp.sum, gain_mask))

  def _lotteryify(k, v):
    if kmatch("**/gain", k):
      return v[gain_mask[k]]

    elif match := kmatch("**/first/Dense/bias", k):
      return v[gain_mask[match.group(1) + "/first/gain"]]
    elif match := kmatch("**/first/Dense/kernel", k):
      return v[:, gain_mask[match.group(1) + "/first/gain"]]

    elif match := kmatch("**/OGDense_*/Dense/bias", k):
      parent = match.group(1)
      k = int(match.group(2))
      return v[gain_mask[f"{parent}/OGDense_{k}/gain"]]
    elif match := kmatch("**/OGDense_*/Dense/kernel", k):
      # If we're an intermediate layer, drop units from the previous layer as
      # input and drop units corresponding to the current layer as output.
      parent = match.group(1)
      k = int(match.group(2))
      in_mask = gain_mask[f"{parent}/OGDense_{k-1}/gain"] if k > 0 else gain_mask[
          f"{parent}/first/gain"]
      out_mask = gain_mask[f"{parent}/OGDense_{k}/gain"]
      # Selecting both masks at once produces an error.
      return v[in_mask, :][:, out_mask]

    elif match := kmatch("**/last/bias", k):
      # Don't drop any units in the last layer.
      return v
    elif match := kmatch("**/last/kernel", k):
      # Only drop inputs to the last layer.
      prev = max(
          int(kmatch("**/OGDense_*/**", k).group(2)) for k in only_gains_final_params_flat.keys()
          if kmatch("**/OGDense_*/**", k))
      return v[gain_mask[match.group(1) + f"/OGDense_{prev}/gain"], :]

    else:
      raise ValueError(f"Unknown key: {k}")

  lottery_init_params = unflatten_params(
      {k: _lotteryify(k, v)
       for k, v in only_gains_final_params_flat.items()})
  print("  lottery params:")
  shapes = tree_map(jnp.shape, flatten_params(lottery_init_params))
  print(shapes)

  # See https://github.com/google/flax/discussions/1555.
  net = make_net([shapes["params/first/gain"][0]] +
                 [shapes[f"params/OGDense_{i}/gain"][0] for i in range(len(gain_params) - 1)])
  train(net,
        init_params=lottery_init_params,
        trainable_predicate=lambda k: True,
        log_prefix="only_gain_lottery")

  print("Training gains+first model...")
  net = make_net([2048] * 6)
  gains_and_first_final_params = train(
      net,
      init_params=net.init(random.PRNGKey(0), jnp.zeros((1, 28 * 28))),
      trainable_predicate=lambda k: kmatch("**/gain", k) or kmatch("**/first/**", k),
      log_prefix="gain_and_first")

  print("Training gains+last model...")
  net = make_net([2048] * 6)
  gains_and_last_final_params = train(
      net,
      init_params=net.init(random.PRNGKey(0), jnp.zeros((1, 28 * 28))),
      trainable_predicate=lambda k: kmatch("**/gain", k) or kmatch("**/last/**", k),
      log_prefix="gain_and_last")

  ##############################################################################
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
