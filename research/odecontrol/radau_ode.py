from operator import itemgetter
from typing import NamedTuple

from jax import jit
from jax import numpy as jnp
from jax import vjp
from jax.experimental import ode
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from scipy import integrate

from research.utils import zeros_like_tree

# TODO: test that our JAX version of RadauDenseOutput agrees with the scipy
# version.

def eval_spline(ta, tb, Q, y_old, t):
  t_local = (t - ta) / (tb - ta)
  return y_old + Q @ jnp.power(t_local, jnp.arange(1, Q.shape[-1] + 1))

class RadauDenseOutput(NamedTuple):
  ts: jnp.array
  Q: jnp.array
  y_old: jnp.array

  def eval(self, t):
    # The first interval is 1, and so on...
    ix = jnp.searchsorted(self.ts, t) - 1
    # In case t is before the beginning or after the end.
    ix = jnp.clip(ix, 0, self.Q.shape[0] - 1)
    return eval_spline(self.ts[ix], self.ts[ix + 1], self.Q[ix], self.y_old[ix], t)

def as_jax_rdo(scipy_rdo):
  """Convert a scipy RadauDenseOutput to a JAX-friendly version."""
  return RadauDenseOutput(ts=jnp.array(scipy_rdo.ts),
                          Q=jnp.array([interp.Q for interp in scipy_rdo.interpolants]),
                          y_old=jnp.array([interp.y_old for interp in scipy_rdo.interpolants]))

def solve_ivp_op(fun, example_y):
  """A collocation-forward, explicit RK-backward solve_ivp.

  `fun` follows the scipy convention: fun(t, y, args)."""
  _, unravel = ravel_pytree(example_y)

  @jit
  def fun_wrapped(t, y, args):
    ydot, _ = ravel_pytree(fun(t, unravel(y), args))
    return ydot

  def fwd(ta, tb, y0, args):
    y0_flat, _ = ravel_pytree(y0)

    solve_ivp_soln = integrate.solve_ivp(lambda t, y: fun_wrapped(t, y, args), (ta, tb),
                                         y0_flat,
                                         method="Radau",
                                         dense_output=True)
    assert solve_ivp_soln.success

    yT = unravel(solve_ivp_soln.y[:, -1])
    y_fn = as_jax_rdo(solve_ivp_soln.sol)

    return yT, y_fn

  @jit
  def bwd_spline_segment(ta, tb, args, Q, y_old, aug_tb):
    """Run the backwards RK on just one segment of the spline."""
    def adj_dynamics(aug, t, args, Q, y_old):
      _, y_bar, _ = aug
      y = unravel(eval_spline(ta, tb, Q, y_old, -t))
      _, vjpfun = vjp(fun, -t, y, args)
      return vjpfun(y_bar)

    adj_path = ode.odeint(adj_dynamics,
                          aug_tb,
                          jnp.array([-tb, -ta]),
                          args,
                          Q,
                          y_old,
                          rtol=1e-3,
                          atol=1e-3)
    return tree_map(itemgetter(-1), adj_path)

  def bwd(args, y_fn, g):
    aug = (jnp.zeros(()), g, zeros_like_tree(args))
    for i in range(y_fn.Q.shape[0])[::-1]:
      # Believe it or not it's faster to pull these out into variables.
      ta = y_fn.ts[i]
      tb = y_fn.ts[i + 1]

      # Believe it or not solve_ivp hands us shit like time steps that are only
      # 1e-16 apart. And those don't play nicely with Runge-Kutta.
      if tb - ta > 1e-8:
        aug = bwd_spline_segment(ta, tb, args, y_fn.Q[i], y_fn.y_old[i], aug)
    (_, _, adj_args) = aug
    return adj_args

  return fwd, bwd

def policy_cost_and_grad(dynamics_fn, cost_fn, policy, example_x):
  def f(_t, y, policy_params):
    _, x = y
    u = policy(policy_params, x)
    return (cost_fn(x, u), dynamics_fn(x, u))

  solve_ivp_fwd, solve_ivp_bwd = solve_ivp_op(f, example_y=(jnp.zeros(()), example_x))

  def run(policy_params, x0, total_time):
    # Run the forward pass.
    y0 = (jnp.zeros(()), x0)
    # t0 = time.time()
    (cost, _), y_fn = solve_ivp_fwd(0.0, total_time, y0, policy_params)
    # print(f"... Forward pass took {time.time() - t0}s")

    # Run the backward pass.
    # t0 = time.time()
    g = (jnp.ones(()), zeros_like_tree(x0))
    g = solve_ivp_bwd(policy_params, y_fn, g)
    # print(f"... Backward pass took {time.time() - t0}s")
    return cost, g

  return run
