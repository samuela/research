from __future__ import annotations

from typing import NamedTuple, TypeVar

_OptState = TypeVar("_OptState")

class Optimizer:
  iteration: int

  def update(self, g) -> Optimizer:
    raise NotImplementedError()

  @property
  def value(self):
    raise NotImplementedError()

def make_optimizer(opt):
  opt_init, opt_update, get_params = opt

  class _Optimizer(NamedTuple, Optimizer):
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
