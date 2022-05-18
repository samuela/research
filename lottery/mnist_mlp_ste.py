import argparse
from pathlib import Path

import jax.numpy as jnp
import optax
import tensorflow as tf
from einops import reduce
from flax.serialization import from_bytes
from flax.training.train_state import TrainState
from jax import jit, random, tree_map, value_and_grad, vmap
from jax.lax import stop_gradient
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

import wandb
from mnist_mlp_run import MLPModel, get_datasets, init_train_state, make_stuff
from utils import (RngPooper, ec2_get_instance_type, flatten_params, rngmix, unflatten_params)

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

def sinkhorn_knopp_projection(A, num_iter=10):
  # We clip to be positive before calling this function.
  A = jnp.maximum(A, 0)
  for _ in range(num_iter):
    # normalize rows
    A = A / reduce(A, "i j -> i 1", "sum")
    # normalize columns
    A = A / reduce(A, "i j -> 1 j", "sum")
  return A

def permute_params_init(rng):
  # Dense_0 Dense_1 Dense_2 Dense_3
  rp = RngPooper(rng)
  return {
      "Dense_0 Dense_1": sinkhorn_knopp_projection(10 + random.uniform(rp.poop(), (512, 512))),
      "Dense_1 Dense_2": sinkhorn_knopp_projection(10 + random.uniform(rp.poop(), (512, 512))),
      "Dense_2 Dense_3": sinkhorn_knopp_projection(10 + random.uniform(rp.poop(), (512, 512))),
  }

def permute_params_apply(permute_params, hardened_permute_params, model_params):

  # See https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html#straight-through-estimator-using-stop-gradient
  def _P(name):
    zero = permute_params[name] - stop_gradient(permute_params[name])
    return stop_gradient(hardened_permute_params[name]) + zero

  invmul = lambda A, B: A.T @ B
  # invmul = jnp.linalg.solve

  P = {
      "Dense_0 Dense_1": _P("Dense_0 Dense_1"),
      "Dense_1 Dense_2": _P("Dense_1 Dense_2"),
      "Dense_2 Dense_3": _P("Dense_2 Dense_3"),
  }

  m = flatten_params(model_params)

  # Dense_0 has a fixed input.
  m["Dense_0/kernel"] = m["Dense_0/kernel"] @ P["Dense_0 Dense_1"]
  m["Dense_0/bias"] = m["Dense_0/bias"] @ P["Dense_0 Dense_1"]

  m["Dense_1/kernel"] = invmul(P["Dense_0 Dense_1"], m["Dense_1/kernel"] @ P["Dense_1 Dense_2"])
  m["Dense_1/bias"] = m["Dense_1/bias"].T @ P["Dense_1 Dense_2"]

  m["Dense_2/kernel"] = invmul(P["Dense_1 Dense_2"], m["Dense_2/kernel"] @ P["Dense_2 Dense_3"])
  m["Dense_2/bias"] = m["Dense_2/bias"].T @ P["Dense_2 Dense_3"]

  # The output of Dense_3 has a fixed order so we don't need to the bias.
  m["Dense_3/kernel"] = invmul(P["Dense_2 Dense_3"], m["Dense_3/kernel"])

  return unflatten_params(m)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-a", type=str, required=True)
  parser.add_argument("--model-b", type=str, required=True)
  parser.add_argument("--test", action="store_true", help="Run in smoke-test mode")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  args = parser.parse_args()

  with wandb.init(
      project="playing-the-lottery",
      entity="skainswo",
      tags=["mnist", "mlp", "straight-through-estimator"],
      # See https://github.com/wandb/client/issues/3672.
      mode="online",
      job_type="analysis",
  ) as wandb_run:
    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.model_a = args.model_a
    config.model_b = args.model_b
    config.test = args.test
    config.seed = args.seed
    config.num_epochs = 50
    config.batch_size = 1000
    config.learning_rate = 1e-2
    # This is the epoch that we pull the model A/B params from.
    config.load_epoch = 49

    model = MLPModel()

    def load_model(filepath):
      with open(filepath, "rb") as fh:
        return from_bytes(init_train_state(random.PRNGKey(0), -1, model), fh.read())

    artifact_a = Path(wandb_run.use_artifact(f"mnist-mlp-weights:{config.model_a}").download())
    artifact_b = Path(wandb_run.use_artifact(f"mnist-mlp-weights:{config.model_b}").download())
    model_a = load_model(artifact_a / f"checkpoint{config.load_epoch}")
    model_b = load_model(artifact_b / f"checkpoint{config.load_epoch}")

    stuff = make_stuff(model)
    train_ds, test_ds = get_datasets(smoke_test_mode=config.test)
    num_train_examples = train_ds["images_u8"].shape[0]
    num_test_examples = test_ds["images_u8"].shape[0]
    assert num_train_examples % config.batch_size == 0
    assert num_test_examples % config.batch_size == 0

    train_loss_a, train_accuracy_a = stuff["dataset_loss_and_accuracy"](model_a.params, train_ds,
                                                                        1000)
    train_loss_b, train_accuracy_b = stuff["dataset_loss_and_accuracy"](model_b.params, train_ds,
                                                                        1000)
    test_loss_a, test_accuracy_a = stuff["dataset_loss_and_accuracy"](model_a.params, test_ds, 1000)
    test_loss_b, test_accuracy_b = stuff["dataset_loss_and_accuracy"](model_b.params, test_ds, 1000)

    print({
        "train_loss_a": train_loss_a,
        "train_accuracy_a": train_accuracy_a,
        "train_loss_b": train_loss_b,
        "train_accuracy_b": train_accuracy_b,
        "test_loss_a": test_loss_a,
        "test_accuracy_a": test_accuracy_a,
        "test_loss_b": test_loss_b,
        "test_accuracy_b": test_accuracy_b,
    })

    baseline_train_loss = 0.5 * (train_loss_a + train_loss_b)

    def harden(permute_params):

      def _harden(A):
        n = A.shape[0]
        ri, ci = linear_sum_assignment(A, maximize=True)
        assert (ri == jnp.arange(len(ri))).all()
        # This is confusing, but indexing the columns onto the rows is actually the correct thing to do
        return jnp.eye(n)[ci, :]

      return {
          "Dense_0 Dense_1": _harden(permute_params["Dense_0 Dense_1"]),
          "Dense_1 Dense_2": _harden(permute_params["Dense_1 Dense_2"]),
          "Dense_2 Dense_3": _harden(permute_params["Dense_2 Dense_3"]),
      }

    @jit
    def batch_eval(permute_params, hardened_permute_params, images_u8, labels):
      model_b_permuted_params = permute_params_apply(permute_params, hardened_permute_params,
                                                     model_b.params)
      interp_params = tree_map(lambda a, b: 0.5 * (a + b), model_a.params, model_b_permuted_params)
      l, num_correct = stuff["batch_eval"](interp_params, images_u8, labels)

      # Makes life easier to know when we're winning. stop_gradient shouldn't be
      # necessary but I'm paranoid.
      l -= stop_gradient(baseline_train_loss)

      return l, {"num_correct": num_correct, "accuracy": num_correct / config.batch_size}

    @jit
    def step(train_state, hardened_permute_params, images_u8, labels):
      (l, metrics), g = value_and_grad(batch_eval,
                                       has_aux=True)(train_state.params, hardened_permute_params,
                                                     images_u8, labels)
      train_state = train_state.apply_gradients(grads=g)

      # Project onto Birkhoff polytope.
      train_state = train_state.replace(
          params=tree_map(sinkhorn_knopp_projection, train_state.params))

      return train_state, {**metrics, "loss": l}

    rng = random.PRNGKey(args.seed)

    tx = optax.sgd(learning_rate=config.learning_rate, momentum=0.9)
    train_state = TrainState.create(apply_fn=None,
                                    params=permute_params_init(rngmix(rng, "init")),
                                    tx=tx)
    # from mnist_mlp_filter_matching import match_filters
    # perm, _ = match_filters(model_a.params, model_b.params)
    # pp = {
    #     "Dense_0 Dense_1": jnp.eye(512)[:, perm["Dense_0 Dense_1"]],
    #     "Dense_1 Dense_2": jnp.eye(512)[:, perm["Dense_1 Dense_2"]],
    #     "Dense_2 Dense_3": jnp.eye(512)[:, perm["Dense_2 Dense_3"]],
    # }
    # train_state = TrainState.create(apply_fn=None, params=pp, tx=tx)

    # model_b_permuted_params = permute_params_apply(train_state.params, harden(train_state.params),
    #                                                model_b.params)
    # interp_params = tree_map(lambda a, b: 0.5 * (a + b), model_a.params, model_b_permuted_params)
    # train_loss_interp, train_accuracy_interp = stuff["dataset_loss_and_accuracy"](interp_params,
    #                                                                               train_ds, 1000)
    # print(f"Interpolated train loss: {train_loss_interp}")
    # print(f"Interpolated train accuracy: {train_accuracy_interp}")

    for epoch in tqdm(range(config.num_epochs)):
      train_data_perm = random.permutation(rngmix(rng, f"epoch-{epoch}"),
                                           num_train_examples).reshape((-1, config.batch_size))
      for i in range(num_train_examples // config.batch_size):
        hardened_pp = harden(train_state.params)
        train_state, metrics = step(train_state, hardened_pp,
                                    train_ds["images_u8"][train_data_perm[i]],
                                    train_ds["labels"][train_data_perm[i]])
        wandb_run.log(metrics)

        if not jnp.isfinite(metrics["loss"]):
          raise ValueError(f"Loss is not finite: {metrics['loss']}")

if __name__ == "__main__":
  main()
