"""See

* https://github.com/facebook/pyre-check/issues/47
* https://github.com/numpy/numpy-stubs
"""
from __future__ import annotations
from typing import Any, Optional, SupportsInt, SupportsFloat, Tuple, TypeVar, Union

# pylint: disable=unused-argument, redefined-builtin

pi: float

Shape = Tuple[int, ...]

class ndarray(SupportsInt, SupportsFloat):
  @property
  def shape(self) -> Shape:
    ...

  def __int__(self) -> int:
    ...

  def __float__(self) -> float:
    ...

  def __getitem__(self, key) -> Any:
    ...

  def __add__(self, other) -> ndarray:
    ...

  def __sub__(self, other) -> ndarray:
    ...

  def __mul__(self, other) -> ndarray:
    ...

  def __rmul__(self, other) -> ndarray:
    ...

  def __div__(self, other) -> ndarray:
    ...

  def __neg__(self) -> ndarray:
    ...

  def __matmul__(self, other) -> ndarray:
    ...

ArrayLike = TypeVar("ArrayLike", int, float, ndarray)

def amax(a: ndarray, axis: Optional[int]) -> ndarray:
  ...

def arange(start: int) -> ndarray:
  ...

def array(object: Any) -> ndarray:
  ...

def broadcast_to(arr: ndarray, shape: Shape) -> ndarray:
  ...

def clip(a: ArrayLike, a_min: Any, a_max: Any) -> ArrayLike:
  ...

def cos(x: ArrayLike) -> ArrayLike:
  ...

def diag(v: ndarray, k: int = 0) -> ndarray:
  ...

def dot(a: ndarray, b: ndarray) -> ndarray:
  ...

def inner(a: ndarray, b: ndarray) -> ndarray:
  ...

def interp(x: ndarray, xp: ndarray, fp: ndarray,
           right: Optional[Any]) -> ndarray:
  ...

def log(x: ArrayLike) -> ArrayLike:
  ...

def sin(x: ArrayLike) -> ArrayLike:
  ...

def sqrt(x: ArrayLike) -> ArrayLike:
  ...

def zeros(shape: Shape, dtype=Any) -> ndarray:
  ...
