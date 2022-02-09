from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.core import freeze
from flax.serialization import from_bytes
from jax import random, tree_map
from tqdm import tqdm

import wandb
from mnist_mlp_run import MLPModel, get_datasets, init_train_state, make_stuff
from permutations import permutify

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

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
