"""random basis BO with derivative info"""
import itertools
import os
import time
from typing import Any, Union

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.objective import ScalarizedObjective
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from jax import grad, jit, random
from jax._src.api import value_and_grad
from jax.experimental import stax
from jax.experimental.stax import Dense, LogSoftmax, Relu
from jax.flatten_util import ravel_pytree

import jax_examples_datasets as datasets
from bo_with_derivatives import GPWithDerivatives

torch.manual_seed(0)

dim_basis = 16
batch_size = 128
train_images, train_labels, test_images, test_labels = datasets.mnist()
num_train = train_images.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)

class RngPooper:
  """A stateful wrapper around stateless random.PRNGKey's."""
  def __init__(self, init_rng):
    self.rng = init_rng

  def poop(self):
    self.rng, rng_key = random.split(self.rng)
    return rng_key

# Data loader stuff...
def data_stream():
  rng = npr.RandomState(0)
  while True:
    perm = rng.permutation(num_train)
    for i in range(num_batches):
      batch_idx = perm[i * batch_size:(i + 1) * batch_size]
      yield train_images[batch_idx], train_labels[batch_idx]

batches = data_stream()

def stax_loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))

init_random_params, predict = stax.serial(Dense(1024), Relu, Dense(1024), Relu, Dense(10),
                                          LogSoftmax)

rp = RngPooper(random.PRNGKey(0))
_, init_params = init_random_params(rp.poop(), (-1, 28 * 28))
init_params_flat, unravel = ravel_pytree(init_params)

# The total number of parameters in the model.
dim_params = jnp.size(init_params_flat)

def flatten(params):
  flat, _ = ravel_pytree(params)
  return flat

def unflatten(params_flat):
  return unravel(params_flat)

# Sample a random orthonormal basis dim_params x dim_basis.
basis, _ = jnp.linalg.qr(random.normal(rp.poop(), (dim_params, dim_basis)))

_obj = lambda x, batch: stax_loss(unflatten(basis @ x), batch)

# def obj_noisy(x):
#   return jit(_obj)(x, next(batches))

def loss_and_grad(x):
  v, g = jit(value_and_grad(_obj))(x, next(batches))
  return jnp.concatenate([jnp.array([v]), g])

jax_to_torch = lambda x: torch.from_numpy(np.asarray(x))
torch_to_jax = lambda x: x.numpy()

num_seed_points = 7
train_x = jax_to_torch([
    flatten(init_random_params(rp.poop(), (-1, 28 * 28))[1]) @ basis for _ in range(num_seed_points)
])
train_y = jax_to_torch([loss_and_grad(torch_to_jax(train_x[i, :])) for i in range(num_seed_points)])

model = GPWithDerivatives(train_X=train_x, train_Y=train_y)
# model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e1))

for it in range(25):
  t0 = time.time()

  # Negate the first element of the objective, since botorch maximizes.
  objective = ScalarizedObjective(torch.tensor([-1.0] + [0.0] * dim_basis))

  # q: number of candidates.
  q = 10

  # candidate_x has shape (q, dim_basis)
  t1 = time.time()
  candidate_x, acq_value = optimize_acqf(
      # UpperConfidenceBound(model, beta=0.1, objective=objective),
      qUpperConfidenceBound(model, beta=0.1, objective=objective),
      bounds=torch.Tensor([[-1] * dim_basis, [1] * dim_basis]),
      q=q,
      # Not sure why these are necessary:
      num_restarts=1,
      raw_samples=1,
  )
  print(f"optimize_acqf took {time.time()-t1} secs")

  # Evaluate candidate point.
  candidate_y = jax_to_torch([loss_and_grad(torch_to_jax(candidate_x[i, :])) for i in range(q)])
  train_x = torch.cat([train_x, candidate_x])
  train_y = torch.cat([train_y, candidate_y])

  # This is currently an error "Cannot yet add fantasy observations to multitask GPs, but this is coming soon!"
  # model = model.condition_on_observations(X=candidate_x, Y=candidate_y)
  model = GPWithDerivatives(train_X=train_x, train_Y=train_y)

  # Train GP...
  mll = ExactMarginalLogLikelihood(model.likelihood, model)
  fit_gpytorch_model(mll)

  print(train_y[:, 0].numpy())
  print(f"Finished iteration {it} in {time.time()-t0} secs.")

  # Plotting...
#   model.eval()

#   fig, ax = plt.subplots(1, 1, figsize=(6, 4))
#   plt.title(f"Bayesian Opt. with derivatives, Iteration {it}")
#   test_x = torch.linspace(-1, 1, steps=100)

#   with torch.no_grad():
#     posterior = model.posterior(test_x)
#     # these are 2 std devs from mean
#     lower, upper = posterior.mvn.confidence_region()

#     ax.plot(test_x.cpu().numpy(),
#             obj(test_x).cpu().numpy(),
#             'r--',
#             label="true, noiseless objective")
#     ax.plot(train_x.cpu().numpy(),
#             train_y[:, 0].cpu().numpy(),
#             'k*',
#             alpha=0.1,
#             label="observations")
#     ax.plot(candidate_x.cpu().numpy(),
#             candidate_y[:, 0].cpu().numpy(),
#             'r*',
#             label="candidate point")
#     ax.plot(test_x.cpu().numpy(), posterior.mean[:, 0].cpu().numpy(), 'b', label="GP posterior")
#     ax.fill_between(test_x.cpu().numpy(),
#                     lower[:, 0].cpu().numpy(),
#                     upper[:, 0].cpu().numpy(),
#                     alpha=0.5)

#   plt.legend(loc="lower left")
#   plt.tight_layout()
#   plt.savefig(f"/tmp/bo_with_derivatives_{it}.jpg")
#   plt.savefig(f"/tmp/bo_with_derivatives_{it}.pdf")

# os.system("ffmpeg -y -r 1 -i /tmp/bo_with_derivatives_%d.jpg bo_with_derivatives.gif")
