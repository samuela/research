from typing import NamedTuple, Tuple

import jax.numpy as jp
from jax import lax, random

NEG_HALF_LOG_TWO_PI = -0.5 * jp.log(2 * jp.pi)

class Distribution:
  @property
  def event_shape(self):
    raise NotImplementedError()

  @property
  def batch_shape(self):
    raise NotImplementedError()

  def sample(self, rng, sample_shape=()):
    raise NotImplementedError()

  def log_prob(self, x):
    raise NotImplementedError()

  def entropy(self):
    raise NotImplementedError()

class Normal(Distribution, NamedTuple):
  loc: jp.array
  scale: jp.array

  @property
  def event_shape(self):
    return ()

  @property
  def batch_shape(self):
    return lax.broadcast_shapes(self.loc.shape, self.scale.shape)

  def sample(self, rng, sample_shape=()):
    return self.loc + self.scale * random.normal(
        rng, shape=sample_shape + self.batch_shape)

  def log_prob(self, x):
    dists = 0.5 * ((x - self.loc) / self.scale)**2.0
    return NEG_HALF_LOG_TWO_PI - jp.log(self.scale) - dists

  def entropy(self):
    return 0.5 - NEG_HALF_LOG_TWO_PI + jp.log(self.scale)

class Uniform(Distribution, NamedTuple):
  minval: jp.array
  maxval: jp.array

  @property
  def event_shape(self):
    return ()

  @property
  def batch_shape(self):
    return lax.broadcast_shapes(self.minval.shape, self.maxval.shape)

  def sample(self, rng, sample_shape=()):
    return self.minval + (self.maxval - self.minval) * random.uniform(
        rng, shape=sample_shape + self.batch_shape)

  def log_prob(self, x):
    return jp.where(
        self.minval <= x <= self.maxval,
        -jp.log(self.maxval - self.minval),
        jp.log(0),
    )

  def entropy(self):
    return jp.log(self.maxval - self.minval)

class Deterministic(Distribution, NamedTuple):
  loc: jp.array
  eps: float = 0.0

  @property
  def event_shape(self):
    return ()

  @property
  def batch_shape(self):
    return self.loc.shape

  def sample(self, _, sample_shape=()):
    return jp.broadcast_to(self.loc, shape=sample_shape + self.batch_shape)

  def log_prob(self, x):
    return jp.where(jp.abs(x - self.loc) <= self.eps, 0.0, jp.log(0))

  def entropy(self):
    return jp.zeros_like(self.loc)

def Independent(reinterpreted_batch_ndims: int):
  # pylint: disable=redefined-outer-name
  class Independent(Distribution, NamedTuple):
    base_distribution: Distribution

    @property
    def event_shape(self):
      return (self.base_distribution.batch_shape[-reinterpreted_batch_ndims:] +
              self.base_distribution.event_shape)

    @property
    def batch_shape(self):
      return self.base_distribution.batch_shape[:-reinterpreted_batch_ndims]

    def sample(self, rng, sample_shape=()):
      return self.base_distribution.sample(rng, sample_shape)

    def log_prob(self, x):
      # Will have shape [sample_shape, base.batch_shape].
      full = self.base_distribution.log_prob(x)
      return jp.sum(full, axis=tuple(range(-reinterpreted_batch_ndims, 0)))

    def entropy(self):
      # Will have shape base.batch_shape.
      full = self.base_distribution.entropy()
      return jp.sum(full, axis=tuple(range(-reinterpreted_batch_ndims, 0)))

  return Independent

def BatchSlice(batch_slice: Tuple):
  def curried(dist: Distribution):
    params_broadcasted = jp.broadcast_arrays(*dist)
    return dist.__class__(*[arr[batch_slice] for arr in params_broadcasted])

  return curried

def DiagMVN(loc: jp.array, scale: jp.array):
  return Independent(1)(Normal(loc, scale))

class MVN(Distribution, NamedTuple):
  loc: jp.array
  cov_cholesky: jp.array

  @property
  def event_shape(self):
    return (self.loc.shape[-1], )

  @property
  def batch_shape(self):
    return self.loc.shape[:-1]

  def sample(self, rng, sample_shape=()):
    z = random.normal(rng, shape=sample_shape + self.event_shape)
    return self.loc + jp.inner(z, self.cov_cholesky)

  def log_prob(self, x):
    # See https://github.com/google/jax/issues/826
    (d, ) = self.event_shape

    # sign should always be 1 since self.cov_cholesky should be PSD. This should
    # have shape [batch_shape].
    _, logdet = jp.linalg.slogdet(self.cov_cholesky)

    # jp.linalg.solve requires the arrays to have compatible ndims.
    delta = x - self.loc
    # delta_extra_ndim = len(delta.shape) - 1
    # cc_extra_ndim = len(self.cov_cholesky.shape) - 2
    # extra_ndim = max(delta_extra_ndim, cc_extra_ndim)
    # delta = delta[(jp.newaxis, ) * (extra_ndim - delta_extra_ndim)]
    # cc = self.cov_cholesky[(jp.newaxis, ) * (extra_ndim - cc_extra_ndim)]

    print(
        f"delta shape = {delta.shape} cov_cholesky.shape = {self.cov_cholesky.shape}"
    )

    broadcast_shape = lax.broadcast_shapes(self.cov_cholesky.shape[:-2],
                                           delta.shape[:-1])
    print(broadcast_shape)
    cc = jp.broadcast_to(self.cov_cholesky, broadcast_shape + (d, d))
    delta = jp.broadcast_to(delta, broadcast_shape + (d, ))
    delta = delta[..., jp.newaxis]

    print(cc.shape)
    print(delta.shape)

    dists = jp.sum(jp.linalg.solve(cc, delta)**2, axis=(-2, -1))
    print(dists)
    return d * NEG_HALF_LOG_TWO_PI - logdet - dists

  def entropy(self):
    raise NotImplementedError()

def Dist(dist):
  def init_fn(_rng, input_shape):
    return input_shape, None

  def apply_fn(_params, inputs, **_):
    return dist(*inputs)

  return init_fn, apply_fn
