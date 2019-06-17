import jax.numpy as jp

from dists import DiagMVN

def normal_kl(dist: DiagMVN):
  """KL(dist || standard normal)"""
  mu, sigma = dist
  return 0.5 * jp.sum(mu**2.0 + sigma**2.0 - 2 * jp.log(sigma) - 1, axis=-1)

def Dampen(init_w: float, epsilon: float = 1e-6):
  def init_fn(_rng, input_shape):
    return input_shape, (jp.array([init_w]), )

  def apply_fn(params, inputs, **_):
    (w, ) = params
    return jp.abs(w) * inputs + epsilon

  return init_fn, apply_fn
