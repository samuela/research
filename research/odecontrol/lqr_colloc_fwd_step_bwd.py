import time
from operator import itemgetter
from typing import NamedTuple

import control
import matplotlib.pyplot as plt
from scipy import integrate

from jax import jit
from jax import numpy as jnp
from jax import random, vjp
from jax.experimental import ode, optimizers, stax
from jax.experimental.stax import Dense, Tanh
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from research import blt
from research.odecontrol.lqr_integrate_cost_trajopt_failure_case import \
    fixed_env
from research.utils import make_optimizer, zeros_like_tree

# import os
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

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

    adj_path = ode.odeint(adj_dynamics, aug_tb, jnp.array([-tb, -ta]), args, Q, y_old)
    return tree_map(itemgetter(-1), adj_path)

  def bwd(args, y_fn, g):
    aug = (jnp.zeros(()), g, zeros_like_tree(args))
    for i in range(y_fn.Q.shape[0])[::-1]:
      aug = bwd_spline_segment(y_fn.ts[i], y_fn.ts[i + 1], args, y_fn.Q[i], y_fn.y_old[i], aug)
    (_, _, adj_args) = aug
    return adj_args

  return fwd, bwd

################################################################################
def policy_cost_and_grad(dynamics_fn, cost_fn, policy, example_x):
  def f(_t, y, policy_params):
    _, x = y
    u = policy(policy_params, x)
    return (cost_fn(x, u), dynamics_fn(x, u))

  solve_ivp_fwd, solve_ivp_bwd = solve_ivp_op(f, example_y=(jnp.zeros(()), example_x))

  def run(policy_params, x0, total_time):
    # Run the forward pass.
    y0 = (jnp.zeros(()), x0)
    t0 = time.time()
    (cost, _), y_fn = solve_ivp_fwd(0.0, total_time, y0, policy_params)
    print(f"      Forward pass took {time.time() - t0}s")

    # Run the backward pass.
    t0 = time.time()
    g = (jnp.ones(()), zeros_like_tree(x0))
    g = solve_ivp_bwd(policy_params, y_fn, g)
    print(f"      Backward pass took {time.time() - t0}s")
    return cost, g

  return run

def main():
  rng = random.PRNGKey(0)
  x_dim = 2
  T = 20.0

  policy_init, policy = stax.serial(
      Dense(64),
      Tanh,
      Dense(x_dim),
  )

  x0 = jnp.ones(x_dim)

  A, B, Q, R, N = fixed_env(x_dim)
  print("System dynamics:")
  print(f"  A = {A}")
  print(f"  B = {B}")
  print(f"  Q = {Q}")
  print(f"  R = {R}")
  print(f"  N = {N}")
  print()

  dynamics_fn = lambda x, u: A @ x + B @ u
  cost_fn = lambda x, u: x.T @ Q @ x + u.T @ R @ u + 2 * x.T @ N @ u

  ### Evaluate LQR solution to get a sense of optimal cost.
  K, _, _ = control.lqr(A, B, Q, R, N)
  K = jnp.array(K)
  opt_policy_cost_fn = policy_cost_and_grad(dynamics_fn,
                                            cost_fn,
                                            lambda KK, x: -KK @ x,
                                            example_x=x0)
  opt_loss, _opt_K_grad = opt_policy_cost_fn(K, x0, T)

  # This is true for longer time horizons, but not true for shorter time
  # horizons due to the LQR solution being an infinite-time solution.
  # assert jnp.allclose(opt_K_grad, 0)

  ### Training loop.
  rng_init_params, rng = random.split(rng)
  _, init_policy_params = policy_init(rng_init_params, (x_dim, ))
  opt = make_optimizer(optimizers.adam(1e-3))(init_policy_params)
  loss_and_grad = policy_cost_and_grad(dynamics_fn, cost_fn, policy, example_x=x0)

  loss_per_iter = []
  elapsed_per_iter = []
  for iteration in range(10000):
    t0 = time.time()
    loss, g = loss_and_grad(opt.value, x0, T)
    opt = opt.update(g)
    elapsed = time.time() - t0

    loss_per_iter.append(loss)
    elapsed_per_iter.append(elapsed)

    print(f"Iteration {iteration}")
    print(f"    excess loss = {loss - opt_loss}")
    print(f"    elapsed = {elapsed}")

  blt.remember({
      "loss_per_iter": loss_per_iter,
      "elapsed_per_iter": elapsed_per_iter,
      "opt_loss": opt_loss
  })

  _, ax1 = plt.subplots()
  ax1.set_xlabel("Iteration")
  ax1.set_ylabel("Cost", color="tab:blue")
  ax1.set_yscale("log")
  ax1.tick_params(axis="y", labelcolor="tab:blue")
  ax1.plot(loss_per_iter, color="tab:blue", label="Total rollout cost")
  plt.axhline(opt_loss, linestyle="--", color="gray")
  ax1.legend(loc="upper left")
  plt.title("Combined fwd-bwd BVP problem")
  blt.show()

if __name__ == "__main__":
  main()
