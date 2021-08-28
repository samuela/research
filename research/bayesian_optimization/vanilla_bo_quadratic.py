"""BO without derivative info"""
import os

import matplotlib.pyplot as plt
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Hartmann
from botorch.utils import standardize
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.manual_seed(0)

def obj(x):
  return -x**2

def obj_noisy(x):
  return obj(x) + 0.1 * torch.randn(x.size())

num_seed_points = 1
train_x = 2 * torch.rand(num_seed_points, 1) - 1
train_y = obj_noisy(train_x)
model = SingleTaskGP(train_X=train_x, train_Y=train_y)
# model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e1))

for it in range(25):
  # Acquisition function...
  candidate_x, acq_value = optimize_acqf(
      UpperConfidenceBound(model, beta=0.1),
      # ExpectedImprovement(model, best_f=torch.max(train_y))
      bounds=torch.Tensor([[-1], [1]]),
      q=1,
      num_restarts=5,
      raw_samples=20,
  )
  candidate_y = obj_noisy(candidate_x)
  train_x = torch.cat([train_x, candidate_x])
  train_y = torch.cat([train_y, candidate_y])
  model = model.condition_on_observations(X=candidate_x, Y=candidate_y)

  # Train GP...
  mll = ExactMarginalLogLikelihood(model.likelihood, model)
  fit_gpytorch_model(mll)

  # Plotting...
  model.eval()

  fig, ax = plt.subplots(1, 1, figsize=(6, 4))
  plt.title(f"Bayesian Opt. without derivatives, Iteration {it}")
  test_x = torch.linspace(-1, 1, steps=100)

  with torch.no_grad():
    posterior = model.posterior(test_x)
    # these are 2 std devs from mean
    lower, upper = posterior.mvn.confidence_region()

    ax.plot(test_x.cpu().numpy(),
            obj(test_x).cpu().numpy(),
            'r--',
            label="true, noiseless objective")
    ax.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k*', alpha=0.1, label="observations")
    ax.plot(candidate_x.cpu().numpy(), candidate_y.cpu().numpy(), 'r*', label="candidate point")
    ax.plot(test_x.cpu().numpy(), posterior.mean.cpu().numpy(), 'b', label="GP posterior")
    ax.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5)

  plt.legend(loc="lower left")
  plt.tight_layout()
  plt.savefig(f"/tmp/vanilla_bo_{it}.jpg")
  plt.savefig(f"/tmp/vanilla_bo_{it}.pdf")

os.system("ffmpeg -y -r 1 -i /tmp/vanilla_bo_%d.jpg vanilla_bo.gif")
