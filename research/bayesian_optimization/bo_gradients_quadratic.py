"""BO with derivative info"""
import os
from typing import Any, Union

import gpytorch
import matplotlib.pyplot as plt
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.objective import ScalarizedObjective
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim import optimize_acqf
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch import settings as gpt_settings
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from torch import Tensor

torch.manual_seed(0)

class GPWithDerivatives(GPyTorchModel, ExactGP):
  def __init__(self, train_X, train_Y):
    d = train_X.shape[-1]
    likelihood = MultitaskGaussianLikelihood(num_tasks=1 + d)
    super(GPWithDerivatives, self).__init__(train_X, train_Y, likelihood)
    self.mean_module = gpytorch.means.ConstantMeanGrad()
    self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=d)
    self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return MultitaskMultivariateNormal(mean_x, covar_x)

  def posterior(self,
                X: Tensor,
                observation_noise: Union[bool, Tensor] = False,
                **kwargs: Any) -> GPyTorchPosterior:
    # need to override this otherwise posterior variances are shot
    with gpt_settings.fast_pred_var(False):
      return super().posterior(X=X, observation_noise=observation_noise, **kwargs)

def obj(x):
  return -x**2

def obj_noisy(x):
  return obj(x) + 0.1 * torch.randn(x.size())

def obj_grad(x):
  return -2 * x

def obj_grad_noisy(x):
  return obj_grad(x) + 0.1 * torch.randn(x.size())

num_seed_points = 1
train_x = 2 * torch.rand(num_seed_points, 1) - 1
# train_x = 0.0 * torch.rand(10, 1)
train_y = torch.cat([obj_noisy(train_x), obj_grad_noisy(train_x)], dim=-1)
model = GPWithDerivatives(train_X=train_x, train_Y=train_y)
# model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e1))

for it in range(25):
  # Acquisition function...
  objective = ScalarizedObjective(torch.tensor([1.0, 0.0]))
  candidate_x, acq_value = optimize_acqf(
      UpperConfidenceBound(model, beta=0.1, objective=objective),
      # ExpectedImprovement(model, best_f=torch.max(train_y))
      bounds=torch.Tensor([[-1], [1]]),
      q=1,
      num_restarts=5,
      raw_samples=20,
  )
  candidate_y = torch.cat([obj_noisy(candidate_x), obj_grad_noisy(candidate_x)], dim=-1)
  train_x = torch.cat([train_x, candidate_x])
  train_y = torch.cat([train_y, candidate_y])

  # This is currently an error "Cannot yet add fantasy observations to multitask GPs, but this is coming soon!"
  # model = model.condition_on_observations(X=candidate_x, Y=candidate_y)
  model = GPWithDerivatives(train_X=train_x, train_Y=train_y)

  # Train GP...
  mll = ExactMarginalLogLikelihood(model.likelihood, model)
  fit_gpytorch_model(mll)

  # Plotting...
  model.eval()

  fig, ax = plt.subplots(1, 1, figsize=(6, 4))
  plt.title(f"Bayesian Opt. with derivatives, Iteration {it}")
  test_x = torch.linspace(-1, 1, steps=100)

  with torch.no_grad():
    posterior = model.posterior(test_x)
    # these are 2 std devs from mean
    lower, upper = posterior.mvn.confidence_region()

    ax.plot(test_x.cpu().numpy(),
            obj(test_x).cpu().numpy(),
            'r--',
            label="true, noiseless objective")
    ax.plot(train_x.cpu().numpy(),
            train_y[:, 0].cpu().numpy(),
            'k*',
            alpha=0.1,
            label="observations")
    ax.plot(candidate_x.cpu().numpy(),
            candidate_y[:, 0].cpu().numpy(),
            'r*',
            label="candidate point")
    ax.plot(test_x.cpu().numpy(), posterior.mean[:, 0].cpu().numpy(), 'b', label="GP posterior")
    ax.fill_between(test_x.cpu().numpy(),
                    lower[:, 0].cpu().numpy(),
                    upper[:, 0].cpu().numpy(),
                    alpha=0.5)

  plt.legend(loc="lower left")
  plt.tight_layout()
  plt.savefig(f"/tmp/bo_with_derivatives_{it}.jpg")
  plt.savefig(f"/tmp/bo_with_derivatives_{it}.pdf")

os.system("ffmpeg -y -r 1 -i /tmp/bo_with_derivatives_%d.jpg bo_with_derivatives.gif")
