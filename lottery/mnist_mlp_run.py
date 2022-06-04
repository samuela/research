"""Train an MLP on MNIST on one random seed. Serialize the model for
interpolation downstream."""
import argparse

import augmax
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, random, value_and_grad, vmap
from tqdm import tqdm

import wandb
from utils import ec2_get_instance_type, rngmix, timeblock

# See https://github.com/tensorflow/tensorflow/issues/53831.

# See https://github.com/google/jax/issues/9454.
tf.config.set_visible_devices([], "GPU")

activation = nn.relu

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
    return loss, {"num_correct": num_correct}

  @jit
  def step(train_state, images_f32, labels):
    (l, info), g = value_and_grad(batch_eval, has_aux=True)(train_state.params, images_f32, labels)
    return train_state.apply_gradients(grads=g), {"batch_loss": l, **info}

  def dataset_loss_and_accuracy(params, dataset, batch_size: int):
    num_examples = dataset["images_u8"].shape[0]
    assert num_examples % batch_size == 0
    num_batches = num_examples // batch_size
    batch_ix = jnp.arange(num_examples).reshape((num_batches, batch_size))
    # Can't use vmap or run in a single batch since that overloads GPU memory.
    losses, infos = zip(*[
        batch_eval(
            params,
            dataset["images_u8"][batch_ix[i, :], :, :, :],
            dataset["labels"][batch_ix[i, :]],
        ) for i in range(num_batches)
    ])
    return (
        jnp.sum(batch_size * jnp.array(losses)) / num_examples,
        sum(x["num_correct"] for x in infos) / num_examples,
    )

  return {
      "normalize_transform": normalize_transform,
      "batch_eval": batch_eval,
      "step": step,
      "dataset_loss_and_accuracy": dataset_loss_and_accuracy
  }

def load_datasets():
  """Return the training and test datasets, unbatched."""
  # See https://www.tensorflow.org/datasets/overview#as_batched_tftensor_batch_size-1.
  train_ds_images_u8, train_ds_labels = tfds.as_numpy(
      tfds.load("mnist", split="train", batch_size=-1, as_supervised=True))
  test_ds_images_u8, test_ds_labels = tfds.as_numpy(
      tfds.load("mnist", split="test", batch_size=-1, as_supervised=True))
  train_ds = {"images_u8": train_ds_images_u8, "labels": train_ds_labels}
  test_ds = {"images_u8": test_ds_images_u8, "labels": test_ds_labels}
  return train_ds, test_ds

def init_train_state(rng, learning_rate, model):
  tx = optax.adam(learning_rate)
  vars = model.init(rng, jnp.zeros((1, 28, 28, 1)))
  return TrainState.create(apply_fn=model.apply, params=vars["params"], tx=tx)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--test", action="store_true", help="Run in smoke-test mode")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  args = parser.parse_args()

  with wandb.init(
      project="playing-the-lottery",
      entity="skainswo",
      tags=["mnist", "mlp"],
      mode="disabled" if args.test else "online",
      job_type="train",
  ) as wandb_run:
    artifact = wandb.Artifact("mnist-mlp-weights", type="model-weights")

    config = wandb.config
    config.ec2_instance_type = ec2_get_instance_type()
    config.test = args.test
    config.seed = args.seed
    config.learning_rate = 0.001
    config.num_epochs = 50
    config.batch_size = 500

    rng = random.PRNGKey(config.seed)

    model = MLPModel()
    stuff = make_stuff(model)

    with timeblock("load_datasets"):
      train_ds, test_ds = load_datasets()
      print("train_ds labels hash", hash(np.array(train_ds["labels"]).tobytes()))
      print("test_ds labels hash", hash(np.array(test_ds["labels"]).tobytes()))

      num_train_examples = train_ds["images_u8"].shape[0]
      num_test_examples = test_ds["images_u8"].shape[0]
      assert num_train_examples % config.batch_size == 0
      print("num_train_examples", num_train_examples)
      print("num_test_examples", num_test_examples)

    train_state = init_train_state(rngmix(rng, "init"), config.learning_rate, model)

    for epoch in tqdm(range(config.num_epochs)):
      infos = []
      with timeblock(f"Epoch"):
        batch_ix = random.permutation(rngmix(rng, f"epoch-{epoch}"), num_train_examples).reshape(
            (-1, config.batch_size))
        for i in range(batch_ix.shape[0]):
          p = batch_ix[i, :]
          images_u8 = train_ds["images_u8"][p, :, :, :]
          labels = train_ds["labels"][p]
          train_state, info = stuff["step"](train_state, images_u8, labels)
          infos.append(info)

      train_loss = sum(config.batch_size * x["batch_loss"] for x in infos) / num_train_examples
      train_accuracy = sum(x["num_correct"] for x in infos) / num_train_examples

      # Evaluate train/test loss/accuracy
      with timeblock("Test set eval"):
        test_loss, test_accuracy = stuff["dataset_loss_and_accuracy"](train_state.params, test_ds,
                                                                      10_000)

      # See https://github.com/wandb/client/issues/3690.
      wandb_run.log({
          "epoch": epoch,
          "train_loss": train_loss,
          "test_loss": test_loss,
          "train_accuracy": train_accuracy,
          "test_accuracy": test_accuracy,
      })

      # With layer width 512, the MLP is 11.2MB per checkpoint.
      with timeblock("model serialization"):
        with artifact.new_file(f"checkpoint{epoch}", mode="wb") as f:
          f.write(flax.serialization.to_bytes(train_state))

    # This will be a no-op when config.test is enabled anyhow, since wandb will
    # be initialized with mode="disabled".
    wandb_run.log_artifact(artifact)

if __name__ == "__main__":
  main()
