import time

import matplotlib.pyplot as plt
import jax.numpy as jp
from jax import jit, random, value_and_grad, vmap
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, FanOut, Softplus
import jax.scipy.special

from research.statistax import BatchSlice, DiagMVN, Independent, MVN, Normal
from research.statistax.stax import DistributionLayer
from .utils import Dampen, normal_kl

theta = 0.5
# eigenvectors are column vectors stacked.
eigenvectors = jp.array([[jp.cos(theta), -jp.sin(theta)],
                         [jp.sin(theta), jp.cos(theta)]])
eigenvalues = jp.array([5, 0.1])
# The scale_tril here isn't actually lower triangular, but it's ok in this case
# because we only sample from it and so it suffices that A @ A.T = covariance.
population_dist = MVN(jp.zeros((2, )),
                      eigenvectors @ jp.diag(jp.sqrt(eigenvalues)))

latent_dim = 64
lam = 1.0
num_iterations = 50000
batch_size = 32
learning_rate = 1e-3
bias_gap = 0.75

def sample_gap_normal(rng, gap: float):
  rng_uniform, rng_bernoulli = random.split(rng)
  u = random.uniform(rng_uniform) * (
      1 - gap) * 0.5 + random.bernoulli(rng_bernoulli) * (1 + gap) * 0.5
  return jax.scipy.special.ndtri(u)

def sample_biased(rng):
  rng_x, rng_y = random.split(rng)
  x = sample_gap_normal(rng_x, bias_gap)
  y = random.normal(rng_y)
  return population_dist.loc + population_dist.scale_tril @ jp.array([x, y])

def demo_plot(rng, num_samples: int):
  rng_population, rng_biased = random.split(rng)
  population_samples = population_dist.sample(rng_population,
                                              sample_shape=(num_samples, ))

  tic = time.time()
  biased_samples = vmap(sample_biased)(random.split(rng_biased, num_samples))
  print(f"biased samples in {time.time() - tic} seconds")

  plt.figure()
  plt.scatter(population_samples[:, 0], population_samples[:, 1])
  plt.title("Population")

  plt.figure()
  plt.scatter(biased_samples[:, 0], biased_samples[:, 1])
  plt.title("Biased sample")
  plt.show()

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
    Dense(128),
    Relu,
    Dense(128),
    Relu,
    Dense(128),
    Relu,
    Dense(128),
    Relu,
    Dense(128),
    Relu,
    FanOut(2),
    stax.parallel(
        Dense(2),
        stax.serial(
            Dense(2),
            Softplus,
        ),
    ),
    DistributionLayer(Normal),
)

def elbo(rng, params, x, decoder_transform, num_mc_samples: int = 1):
  encoder_params, decoder_params = params
  approx_posterior = encoder(encoder_params, x)

  # The z's to evaluate sampled from the approximate posterior.
  zs = approx_posterior.sample(rng, sample_shape=(num_mc_samples, ))

  # Calculates log p(x | z) for a single z.
  loglik = lambda z: decoder_transform(decoder(decoder_params, z)).log_prob(x)

  # Average over the z samples used to estimate the expectation.
  return jp.mean(vmap(loglik)(zs)) - normal_kl(
      approx_posterior.base_distribution)

opt_init, opt_update, get_params = optimizers.adam(learning_rate)

@jit
def step(rng, i, opt_state):
  rng_population, rng_biased = random.split(rng)
  population_batch = population_dist.sample(rng_population,
                                            sample_shape=(batch_size, ))
  biased_batch = vmap(sample_biased)(random.split(rng_biased, batch_size))

  def joint_elbo_batch(params):
    joint_enc_params, _, dec_params = params
    elbo_one = lambda rng, x: -elbo(rng, (joint_enc_params, dec_params), x,
                                    Independent(1))
    rngs = random.split(rng, batch_size)
    return jp.mean(vmap(elbo_one, in_axes=(0, 0))(rngs, biased_batch))

  def marginal_elbo_batch(params, dim: int):
    _, marginal_enc_params, dec_params = params
    elbo_one = lambda rng, x: -elbo(rng, (marginal_enc_params[dim], dec_params
                                          ), x, BatchSlice((..., dim)))
    rngs = random.split(rng, batch_size)
    return jp.mean(
        vmap(elbo_one, in_axes=(0, 0))(rngs, population_batch[..., dim]))

  def full_elbo(params):
    return joint_elbo_batch(params) + 0.5 * lam * (
        marginal_elbo_batch(params, 0) + marginal_elbo_batch(params, 1))

  loss, g = value_and_grad(full_elbo)(get_params(opt_state))
  return loss, g, opt_update(i, g, opt_state)

def plot_samples(rng, iteration: int, opt_state, num_samples: int = 1024):
  population_samples = population_dist.sample(rng,
                                              sample_shape=(num_samples, ))

  biased_samples = vmap(sample_biased)(random.split(rng, num_samples))

  _, _, decoder_params = get_params(opt_state)
  prior = DiagMVN(jp.zeros((latent_dim, )), jp.ones((latent_dim, )))
  z_rng, x_rng = random.split(rng)
  zs = prior.sample(z_rng, sample_shape=(num_samples, ))
  xs = vmap(lambda z: decoder(decoder_params, z))(zs).sample(x_rng)

  plt.figure(figsize=(12, 4))
  ax = plt.subplot(1, 3, 1)
  plt.scatter(population_samples[:, 0], population_samples[:, 1], alpha=0.5)
  plt.legend(["Population"])

  plt.subplot(1, 3, 2, sharex=ax, sharey=ax)
  plt.scatter(biased_samples[:, 0], biased_samples[:, 1], alpha=0.5)
  plt.legend(["Biased sample"])

  plt.subplot(1, 3, 3, sharex=ax, sharey=ax)
  plt.scatter(xs[:, 0], xs[:, 1], alpha=0.5)
  plt.legend(["VAE samples"])

  plt.suptitle(f"Iteration {iteration}, lambda = {lam}")
  plt.show()

def main(rng):
  encoder_init_rng, decoder_init_rng, rng = random.split(rng, 3)
  _, init_joint_encoder_params = encoder_init(encoder_init_rng, (2, ))
  _, init_x_encoder_params = encoder_init(encoder_init_rng, (1, ))
  _, init_y_encoder_params = encoder_init(encoder_init_rng, (1, ))
  _, init_decoder_params = decoder_init(decoder_init_rng, (latent_dim, ))

  opt_state = opt_init((init_joint_encoder_params,
                        [init_x_encoder_params,
                         init_y_encoder_params], init_decoder_params))

  rngs = random.split(rng, num_iterations)
  for i in range(num_iterations):
    tic = time.time()
    # sample_rng, elbo_rng = random.split(rngs[i])
    # batch = population_dist.sample(sample_rng, sample_shape=(batch_size, ))
    loss_val, _, opt_state = step(rngs[i], i, opt_state)
    print(
        f"Iteration {i}\tELBO: {-1 * loss_val}\tduration: {time.time() - tic}")

    if (i + 1) % 1000 == 0:
      plot_samples(rng, i, opt_state)

if __name__ == "__main__":
  # demo_plot(random.PRNGKey(0), 1000)
  main(random.PRNGKey(0))
