"""See

* https://github.com/facebook/pyre-check/issues/47
* https://github.com/numpy/numpy-stubs
"""

from typing import Any, Optional

# pylint: disable=unused-argument, redefined-builtin

def array(object: Any) -> Any:
  ...

def clip(a: Any, a_min: Any, a_max: Any) -> Any:
  ...

def amax(a: Any, axis: Optional[int]) -> Any:
  ...

def dot(a: Any, b: Any) -> Any:
  ...

def arange(start: int) -> Any:
  ...

def interp(x: Any, xp: Any, fp: Any, right: Optional[Any]) -> Any:
  ...
