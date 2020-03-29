from __future__ import annotations
from typing import NamedTuple, TypeVar, Generic
from jax.nn.initializers import glorot_normal
from jax import random
import jax.numpy as jp

_OptState = TypeVar("_OptState")

class Optimizer:
  def update(self, g) -> Optimizer:
    raise NotImplementedError()

  @property
  def value(self):
    raise NotImplementedError()

  @property
  def iteration(self):
    raise NotImplementedError()

def make_optimizer(opt):
  opt_init, opt_update, get_params = opt

  class _Optimizer(Generic[_OptState], NamedTuple, Optimizer):
    iteration: int
    opt_state: _OptState

    def update(self, g) -> Optimizer:
      return _Optimizer(
          iteration=self.iteration + 1,
          opt_state=opt_update(self.iteration, g, self.opt_state),
      )

    @property
    def value(self):
      return get_params(self.opt_state)

  def start(init_params):
    return _Optimizer(iteration=0, opt_state=opt_init(init_params))

  return start

def DenseNoBias(out_dim, W_init=glorot_normal()):
  """Layer constructor function for a dense (fully-connected) layer but without
  any bias term."""
  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim, )
    W = W_init(rng, (input_shape[-1], out_dim))
    return output_shape, W

  def apply_fun(W, inputs, **_kwargs):
    return inputs @ W

  return init_fun, apply_fun

def random_psd(rng, n):
  x = random.normal(rng, shape=(n, n))
  return x.T @ x

def random_orthonormal(rng, n):
  u, _, vh = jp.linalg.svd(random.normal(rng, shape=(n, n)))
  return u @ vh
