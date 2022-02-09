from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.core import freeze
from flax.serialization import from_bytes
from jax import random, tree_map
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

import wandb
from mnist_mlp_run import MLPModel, get_datasets, init_train_state, make_stuff
from utils import (RngPooper, flatten_params, kmatch, timeblock, unflatten_params)

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

def cosine_similarity(X, Y):
  # X: (m, d)
  # Y: (n, d)
  # return: (m, n)
  return (X @ Y.T) / jnp.linalg.norm(X, axis=-1).reshape((-1, 1)) / jnp.linalg.norm(Y, axis=-1)

def permutify(params1, params2):
  """Permute the parameters of params2 to match params1 as closely as possible.
  Returns the permuted version of params2. Only works on sequences of Dense
  layers for now."""
  p1f = flatten_params(params1)
  p2f = flatten_params(params2)

  p2f_new = {**p2f}
  num_layers = max(int(kmatch("params/Dense_*/**", k).group(1)) for k in p1f.keys())
  # range is [0, num_layers), so we're safe here since we don't want to be
  # reordering the output of the last layer.
  for layer in range(num_layers):
    # Maximize since we're dealing with similarities, not distances.
    ri, ci = linear_sum_assignment(cosine_similarity(p1f[f"params/Dense_{layer}/kernel"].T,
                                                     p2f_new[f"params/Dense_{layer}/kernel"].T),
                                   maximize=True)
    assert (ri == jnp.arange(len(ri))).all()

    p2f_new = {
        **p2f_new, f"params/Dense_{layer}/kernel": p2f_new[f"params/Dense_{layer}/kernel"][:, ci],
        f"params/Dense_{layer}/bias": p2f_new[f"params/Dense_{layer}/bias"][ci],
        f"params/Dense_{layer+1}/kernel": p2f_new[f"params/Dense_{layer+1}/kernel"][ci, :]
    }

  return unflatten_params(p2f_new)

def test_cosine_similarity():
  rp = RngPooper(random.PRNGKey(0))

  for _ in range(10):
    X = random.normal(rp.poop(), (3, 5))
    Y = random.normal(rp.poop(), (7, 5))
    assert jnp.allclose(1 - cosine_similarity(X, Y), cdist(X, Y, metric="cosine"))

def test_permutify():
  rp = RngPooper(random.PRNGKey(0))

  class Model(nn.Module):

    @nn.compact
    def __call__(self, x):
      x = nn.Dense(1024, bias_init=nn.initializers.normal(stddev=1.0))(x)
      x = nn.relu(x)
      x = nn.Dense(1024, bias_init=nn.initializers.normal(stddev=1.0))(x)
      x = nn.relu(x)
      x = nn.Dense(10)(x)
      x = nn.log_softmax(x)
      return x

  model = Model()
  p1 = model.init(rp.poop(), jnp.zeros((1, 28 * 28)))
  p2 = model.init(rp.poop(), jnp.zeros((1, 28 * 28)))
  # print(tree_map(jnp.shape, flatten_params(p1)))

  new_p2 = permutify(p1, p2)

  # Test that the model is the same after permutation.
  random_input = random.normal(rp.poop(), (128, 28 * 28))
  # print(jnp.max(jnp.abs(model.apply(p2, random_input) - model.apply(new_p2, random_input))))
  assert ((jnp.abs(model.apply(p2, random_input) - model.apply(new_p2, random_input))) < 1e-5).all()

def plot_interp_loss(epoch, lambdas, train_loss_interp_naive, test_loss_interp_naive,
                     train_loss_interp_clever, test_loss_interp_clever):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(lambdas,
          train_loss_interp_naive,
          linestyle="dashed",
          color="tab:blue",
          alpha=0.5,
          linewidth=2,
          label="Train, naïve interp.")
  ax.plot(lambdas,
          test_loss_interp_naive,
          linestyle="dashed",
          color="tab:orange",
          alpha=0.5,
          linewidth=2,
          label="Test, naïve interp.")
  ax.plot(lambdas,
          train_loss_interp_clever,
          linestyle="solid",
          color="tab:blue",
          linewidth=2,
          label="Train, permuted interp.")
  ax.plot(lambdas,
          test_loss_interp_clever,
          linestyle="solid",
          color="tab:orange",
          linewidth=2,
          label="Test, permuted interp.")
  ax.set_xlabel("$\lambda$")
  ax.set_xticks([0, 1])
  ax.set_xticklabels(["Model $A$", "Model $B$"])
  ax.set_ylabel("Loss")
  # TODO label x=0 tick as \theta_1, and x=1 tick as \theta_2
  ax.set_title(f"Loss landscape between the two models (epoch {epoch})")
  ax.legend(loc="upper right", framealpha=0.5)
  fig.tight_layout()
  return fig

def plot_interp_acc(epoch, lambdas, train_acc_interp_naive, test_acc_interp_naive,
                    train_acc_interp_clever, test_acc_interp_clever):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(lambdas,
          train_acc_interp_naive,
          linestyle="dashed",
          color="tab:blue",
          alpha=0.5,
          linewidth=2,
          label="Train, naïve interp.")
  ax.plot(lambdas,
          test_acc_interp_naive,
          linestyle="dashed",
          color="tab:orange",
          alpha=0.5,
          linewidth=2,
          label="Test, naïve interp.")
  ax.plot(lambdas,
          train_acc_interp_clever,
          linestyle="solid",
          color="tab:blue",
          linewidth=2,
          label="Train, permuted interp.")
  ax.plot(lambdas,
          test_acc_interp_clever,
          linestyle="solid",
          color="tab:orange",
          linewidth=2,
          label="Test, permuted interp.")
  ax.set_xlabel("$\lambda$")
  ax.set_xticks([0, 1])
  ax.set_xticklabels(["Model $A$", "Model $B$"])
  ax.set_ylabel("Accuracy")
  # TODO label x=0 tick as \theta_1, and x=1 tick as \theta_2
  ax.set_title(f"Accuracy between the two models (epoch {epoch})")
  ax.legend(loc="lower right", framealpha=0.5)
  fig.tight_layout()
  return fig

if __name__ == "__main__":
  with timeblock("Tests"):
    test_cosine_similarity()
    test_permutify()

  epoch = 49
  model = MLPModel()

  def load_checkpoint(run, epoch):
    f = wandb.restore(f"checkpoint_{epoch}", run)
    with open(f.name, "rb") as f:
      _, ret = from_bytes((0, init_train_state(random.PRNGKey(0), -1, model)), f.read())
    Path(f.name).unlink()
    return ret

  model_a = load_checkpoint("skainswo/playing-the-lottery/2iwebdo3", epoch)
  model_b = load_checkpoint("skainswo/playing-the-lottery/2vu72civ", epoch)

  lambdas = jnp.linspace(0, 1, num=10)
  train_loss_interp_clever = []
  test_loss_interp_clever = []
  train_acc_interp_clever = []
  test_acc_interp_clever = []

  train_loss_interp_naive = []
  test_loss_interp_naive = []
  train_acc_interp_naive = []
  test_acc_interp_naive = []

  stuff = make_stuff(model)
  train_ds, test_ds = get_datasets(test_mode=False)
  num_train_examples = train_ds.cardinality().numpy()
  num_test_examples = test_ds.cardinality().numpy()
  # Might as well use the larget batch size that we can fit into memory here.
  train_ds_batched = tfds.as_numpy(train_ds.batch(2048))
  test_ds_batched = tfds.as_numpy(test_ds.batch(2048))
  for lam in tqdm(lambdas):
    # TODO make this look like the permuted version below
    naive_p = tree_map(lambda a, b: lam * a + (1 - lam) * b, model_a.params, model_b.params)
    train_loss_interp_naive.append(stuff.dataset_loss(naive_p, train_ds_batched))
    test_loss_interp_naive.append(stuff.dataset_loss(naive_p, test_ds_batched))
    train_acc_interp_naive.append(
        stuff.dataset_total_correct(naive_p, train_ds_batched) / num_train_examples)
    test_acc_interp_naive.append(
        stuff.dataset_total_correct(naive_p, test_ds_batched) / num_test_examples)

    b2 = permutify({"params": model_a.params}, {"params": model_b.params})
    clever_p = tree_map(lambda a, b: lam * a + (1 - lam) * b, freeze({"params": model_a.params}),
                        b2)
    train_loss_interp_clever.append(stuff.dataset_loss(clever_p["params"], train_ds_batched))
    test_loss_interp_clever.append(stuff.dataset_loss(clever_p["params"], test_ds_batched))
    train_acc_interp_clever.append(
        stuff.dataset_total_correct(clever_p["params"], train_ds_batched) / num_train_examples)
    test_acc_interp_clever.append(
        stuff.dataset_total_correct(clever_p["params"], test_ds_batched) / num_test_examples)

  assert len(lambdas) == len(train_loss_interp_naive)
  assert len(lambdas) == len(test_loss_interp_naive)
  assert len(lambdas) == len(train_acc_interp_naive)
  assert len(lambdas) == len(test_acc_interp_naive)
  assert len(lambdas) == len(train_loss_interp_clever)
  assert len(lambdas) == len(test_loss_interp_clever)
  assert len(lambdas) == len(train_acc_interp_clever)
  assert len(lambdas) == len(test_acc_interp_clever)

  print("Plotting...")
  fig = plot_interp_loss(epoch, lambdas, train_loss_interp_naive, test_loss_interp_naive,
                         train_loss_interp_clever, test_loss_interp_clever)
  plt.savefig(f"mnist_mlp_interp_loss_epoch{epoch}.png", dpi=300)
  plt.close(fig)

  fig = plot_interp_acc(epoch, lambdas, train_acc_interp_naive, test_acc_interp_naive,
                        train_acc_interp_clever, test_acc_interp_clever)
  plt.savefig(f"mnist_mlp_interp_accuracy_epoch{epoch}.png", dpi=300)
  plt.close(fig)
