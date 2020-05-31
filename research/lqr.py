from jax import random
import jax.numpy as jp
from research.utils import random_psd

def random_env(rng, n):
  rngA, rngB, rngQ, rngR = random.split(rng, 4)
  A = -1 * random_psd(rngA, n)
  B = random.normal(rngB, (n, n))
  Q = random_psd(rngQ, n) + 0.1 * jp.eye(n)
  R = random_psd(rngR, n) + 0.1 * jp.eye(n)
  N = jp.zeros((n, n))
  return A, B, Q, R, N

def lqr_continuous_time_infinite_horizon(A, B, Q, R, N):
  # Take the last dimension, in case we try to do some kind of broadcasting
  # thing in the future.
  x_dim = A.shape[-1]

  # pylint: disable=line-too-long
  # See https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator#Infinite-horizon,_continuous-time_LQR.
  A1 = A - B @ jp.linalg.solve(R, N.T)
  Q1 = Q - N @ jp.linalg.solve(R, N.T)

  # See https://en.wikipedia.org/wiki/Algebraic_Riccati_equation#Solution.
  H = jp.block([[A1, -B @ jp.linalg.solve(R, B.T)], [-Q1, -A1]])
  eigvals, eigvectors = jp.linalg.eig(H)

  # For large-ish systems (eg x_dim = 7), sometimes we find some values that
  # have an imaginary component. That's an unfortunate consequence of the
  # numerical instability in the eigendecomposition. Still,
  # assert (eigvals.imag == jp.zeros_like(eigvals, dtype=jp.float32)).all()
  # assert (eigvectors.imag == jp.zeros_like(eigvectors, dtype=jp.float32)).all()

  # Now it should be safe to take out only the real components.
  eigvals = eigvals.real
  eigvectors = eigvectors.real

  argsort = jp.argsort(eigvals)
  ix = argsort[:x_dim]
  U = eigvectors[:, ix]
  P = U[x_dim:, :] @ jp.linalg.inv(U[:x_dim, :])

  K = jp.linalg.solve(R, (B.T @ P + N.T))
  return K, P, eigvals[ix]

################################################################################
def _test_lqr1(n):
  # pylint: disable=import-outside-toplevel
  import control
  from jax.tree_util import tree_multimap

  A = jp.eye(n)
  B = jp.eye(n)
  Q = jp.eye(n)
  R = jp.eye(n)
  N = jp.zeros((n, n))

  actual = lqr_continuous_time_infinite_horizon(A, B, Q, R, N)
  expected = control.lqr(A, B, Q, R, N)
  assert tree_multimap(jp.allclose, actual, expected)

def _test_lqr2(n):
  # pylint: disable=import-outside-toplevel
  import control
  from jax.tree_util import tree_multimap

  A = jp.zeros((n, n))
  B = jp.eye(n)
  Q = jp.eye(n)
  R = jp.eye(n)
  N = jp.zeros((n, n))

  actual = lqr_continuous_time_infinite_horizon(A, B, Q, R, N)
  expected = control.lqr(A, B, Q, R, N)
  assert tree_multimap(jp.allclose, actual, expected)

def _test_lqr_random(rng, n):
  # pylint: disable=import-outside-toplevel
  import control
  from jax.tree_util import tree_multimap

  A, B, Q, R, N = random_env(rng, n)
  actual = lqr_continuous_time_infinite_horizon(A, B, Q, R, N)
  expected = control.lqr(A, B, Q, R, N)
  assert tree_multimap(jp.allclose, actual, expected)

def _run_all_tests():
  for n in range(1, 10):
    _test_lqr1(n)
    _test_lqr2(n)
    _test_lqr_random(random.PRNGKey(n), n)

if __name__ == "__main__":
  _run_all_tests()
