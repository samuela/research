from typing import Any, Union

import gpytorch
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch import settings as gpt_settings
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.models import ExactGP
from torch import Tensor


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
