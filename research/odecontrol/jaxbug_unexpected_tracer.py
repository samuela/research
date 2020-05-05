from jax import random
from jax import vmap
import jax.numpy as jp
from jax.experimental import ode

if __name__ == "__main__":
  rng = random.PRNGKey(0)

  ### Set up the problem/environment
  # xdot = Ax + Bu
  # u = - Kx
  # cost = xQx + uRu + 2xNu

  A = jp.eye(2)
  B = jp.eye(2)
  Q = jp.eye(2)
  R = jp.eye(2)
  N = jp.zeros((2, 2))

  dynamics_fn = lambda x, u: A @ x + B @ u
  cost_fn = lambda x, u: x.T @ Q @ x + u.T @ R @ u + 2 * x.T @ N @ u
  policy = lambda x: -x

  def ofunc(y, _t):
    x = y[1:]
    u = policy(x)
    return jp.concatenate((jp.expand_dims(cost_fn(x, u), axis=0), dynamics_fn(x, u)))

  def evally(x0):
    # Zero is necessary for some reason...
    t = jp.array([0.0, 1.0])
    y0 = jp.concatenate((jp.zeros((1, )), x0))
    yT = ode.odeint(ofunc, y0, t)
    return yT[1, 0]

  x0_eval = random.normal(rng, (1000, 2))
  vmap(evally)(x0_eval)
