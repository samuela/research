import time

import matplotlib.pyplot as plt
import jax.numpy as jp
from jax import jit, random, value_and_grad, vmap
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, FanOut, Softplus

from research.statistax import DiagMVN, MVN
from research.statistax.stax import DistributionLayer
from .utils import Dampen, normal_kl

theta = 0.5
# eigenvectors are column vectors stacked.
eigenvectors = jp.array([[jp.cos(theta), -jp.sin(theta)],
                         [jp.sin(theta), jp.cos(theta)]])
eigenvalues = jp.array([5, 0.1])
true_dist = MVN(jp.zeros((2, )), eigenvectors @ jp.diag(jp.sqrt(eigenvalues)))

latent_dim = 2
num_iterations = 5000
batch_size = 1024
learning_rate = 1e-2

encoder_init, encoder = stax.serial(
    Dense(32),
    Relu,
    FanOut(2),
    stax.parallel(
        Dense(latent_dim),
        stax.serial(
            Dense(latent_dim),
            Softplus,
            Dampen(0.1, 1e-6),
        ),
    ),
    DistributionLayer(DiagMVN),
)

decoder_init, decoder = stax.serial(
    FanOut(2),
    stax.parallel(
        Dense(2),
        stax.serial(
            Dense(2),
            Softplus,
        ),
    ),
    DistributionLayer(DiagMVN),
)

def elbo(rng, params, x, num_mc_samples: int = 1):
  encoder_params, decoder_params = params
  approx_posterior = encoder(encoder_params, x)

  # The z's to evaluate sampled from the approximate posterior.
  zs = approx_posterior.sample(rng, sample_shape=(num_mc_samples, ))

  # Calculates log p(x | z) for a single z.
  loglik = lambda z: decoder(decoder_params, z).log_prob(x)

  # Average over the z samples used to estimate the expectation. Note that we
  # need to pull the normal distribution out of the Independent one.
  return jp.mean(vmap(loglik)(zs)) - normal_kl(
      approx_posterior.base_distribution)

opt_init, opt_update, get_params = optimizers.adam(learning_rate)

@jit
def step(rng, i: int, opt_state, batch):
  def elbo_batch(params):
    elbo_one = lambda rng, x: -elbo(rng, params, x, num_mc_samples=1)
    rngs = random.split(rng, batch.shape[0])
    return jp.mean(vmap(elbo_one, in_axes=(0, 0))(rngs, batch))

  loss, g = value_and_grad(elbo_batch)(get_params(opt_state))
  return loss, g, opt_update(i, g, opt_state)

def plot_samples(rng, iteration, opt_state, num_samples: int = 1024):
  _, decoder_params = get_params(opt_state)
  prior = DiagMVN(jp.zeros((latent_dim, )), jp.ones((latent_dim, )))
  z_rng, x_rng = random.split(rng)
  zs = prior.sample(z_rng, sample_shape=(num_samples, ))
  xs = vmap(lambda z: decoder(decoder_params, z))(zs).sample(x_rng)

  true_samples = true_dist.sample(rng, sample_shape=(num_samples, ))

  plt.figure()
  plt.scatter(true_samples[:, 0], true_samples[:, 1])
  plt.scatter(xs[:, 0], xs[:, 1])
  plt.legend(["original dist.", "VAE"])
  plt.title(f"Iteration {iteration}")
  plt.show()

def main(rng):
  encoder_init_rng, decoder_init_rng, rng = random.split(rng, 3)
  _, init_encoder_params = encoder_init(encoder_init_rng, (2, ))
  _, init_decoder_params = decoder_init(decoder_init_rng, (latent_dim, ))
  opt_state = opt_init((init_encoder_params, init_decoder_params))

  rngs = random.split(rng, num_iterations)
  elbo_per_iter = []
  for i in range(num_iterations):
    tic = time.time()
    sample_rng, elbo_rng = random.split(rngs[i])
    batch = true_dist.sample(sample_rng, sample_shape=(batch_size, ))
    loss_val, _, opt_state = step(elbo_rng, i, opt_state, batch)
    print(
        f"Iteration {i}\tELBO: {-1 * loss_val}\telapsed: {time.time() - tic}")

    elbo_per_iter.append(-loss_val.item())

    if (i + 1) % 1000 == 0:
      plot_samples(rng, i, opt_state)

  plt.figure()
  plt.plot(elbo_per_iter)
  plt.xlabel("Iteration")
  plt.ylabel("ELBO")
  plt.show()

if __name__ == "__main__":
  main(random.PRNGKey(0))
