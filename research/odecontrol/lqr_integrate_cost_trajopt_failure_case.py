"""This is an attempt at naively using a neural ODE for trajectory optimization
over a finite time horizon. This script shows that it fails miserably due to
issues recovering the initial conditions in reverse pass."""

from operator import itemgetter
import time
import control
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
import jax.numpy as jp
from jax.experimental import stax
from jax.experimental import ode
from jax.experimental import optimizers
from jax.experimental.stax import Dense
from jax.experimental.stax import Tanh
from jax.tree_util import tree_map
from research.utils import make_optimizer
from research.utils import random_psd
from research.utils import zeros_like_tree
from research import blt

def fixed_env(n):
  A = 0 * jp.eye(n)
  # A = jp.diag(jp.array([-1.0, 1.0]))
  B = jp.eye(n)
  Q = jp.eye(n)
  R = jp.eye(n)
  N = jp.zeros((n, n))
  return A, B, Q, R, N

def random_env(rng):
  rngA, rngB, rngQ, rngR = random.split(rng, 4)
  A = -1 * random_psd(rngA, 2)
  B = random.normal(rngB, (2, 2))
  Q = random_psd(rngQ, 2) + 0.1 * jp.eye(2)
  R = random_psd(rngR, 2) + 0.1 * jp.eye(2)
  N = jp.zeros((2, 2))
  return A, B, Q, R, N

def policy_integrate_cost(dynamics_fn, cost_fn, gamma):
  # Specialize to the environment.

  def eval_policy(policy):
    # Specialize to the policy.

    def ofunc(y, t, policy_params):
      _, x = y
      u = policy(policy_params, x)
      return ((gamma**t) * cost_fn(x, u), dynamics_fn(x, u))

    def eval_from_x0(policy_params, x0, total_time):
      # Zero is necessary for some reason...
      t = jp.array([0.0, total_time])
      y0 = (jp.zeros(()), x0)
      odeint_kwargs = {"mxstep": 1e6}
      y_fwd = ode.odeint(ofunc, y0, t, policy_params, **odeint_kwargs)
      yT = tree_map(itemgetter(1), y_fwd)

      # This is similar but not exactly the same as the place that the rev-mode
      # solution since the step sizes can vary when using all the other
      # parameters.
      y_bwd = ode.odeint(lambda y, t, *args: tree_map(jp.negative, ofunc(y, -t, *args)), yT,
                         -t[::-1], policy_params, **odeint_kwargs)
      y0_bwd = tree_map(itemgetter(1), y_bwd)

      return y0, yT, y0_bwd

    return eval_from_x0

  return eval_policy

def main():
  total_time = 20.0
  gamma = 1.0
  x_dim = 2
  rng = random.PRNGKey(0)

  x0 = jp.array([2.0, 1.0])

  ### Set up the problem/environment
  # xdot = Ax + Bu
  # u = - Kx
  # cost = xQx + uRu + 2xNu
  A, B, Q, R, N = fixed_env(x_dim)
  dynamics_fn = lambda x, u: A @ x + B @ u
  cost_fn = lambda x, u: x.T @ Q @ x + u.T @ R @ u + 2 * x.T @ N @ u
  policy_loss = policy_integrate_cost(dynamics_fn, cost_fn, gamma)

  ### Solve the Riccatti equation to get the infinite-horizon optimal solution.
  K, _, _ = control.lqr(A, B, Q, R, N)
  K = jp.array(K)

  t0 = time.time()
  _, (opt_cost_fwd, opt_xT_fwd), (opt_cost_bwd,
                                  opt_x0_bwd) = policy_loss(lambda _, x: -K @ x)(None, x0,
                                                                                 total_time)
  print(f"opt_cost_fwd = {opt_cost_fwd}, opt_cost_bwd = {opt_cost_bwd} in {time.time() - t0}s")
  print(opt_xT_fwd)
  print(opt_x0_bwd)
  print(f"l2 error: {jp.sqrt(jp.sum((x0 - opt_x0_bwd)**2))}")

  ### Set up the learned policy model.
  policy_init, policy = stax.serial(
      Dense(64),
      Tanh,
      Dense(x_dim),
  )

  rng_init_params, rng = random.split(rng)
  _, init_policy_params = policy_init(rng_init_params, (x_dim, ))
  opt = make_optimizer(optimizers.adam(1e-3))(init_policy_params)
  runny_run = jit(policy_loss(policy))

  ### Main optimization loop.
  cost_T_fwd_per_iter = []
  xT_fwd_per_iter = []
  cost_0_bwd_per_iter = []
  x0_bwd_per_iter = []
  for i in range(5000):
    t0 = time.time()
    (y0_fwd, yT_fwd, y0_bwd), vjp = jax.vjp(runny_run, opt.value, x0, total_time)
    cost_T_fwd, xT_fwd = yT_fwd
    cost_0_bwd, x0_bwd = y0_bwd

    yT_fwd_bar = (jp.ones(()), jp.zeros_like(x0))
    g, _, _ = vjp((zeros_like_tree(y0_fwd), yT_fwd_bar, zeros_like_tree(y0_bwd)))
    opt = opt.update(g)

    cost_T_fwd_per_iter.append(cost_T_fwd)
    xT_fwd_per_iter.append(xT_fwd)
    cost_0_bwd_per_iter.append(cost_0_bwd)
    x0_bwd_per_iter.append(x0_bwd)

    print(f"Episode {i}:")
    print(f"  excess fwd cost = {cost_T_fwd - opt_cost_fwd}")
    print(f"  bwd cost        = {cost_0_bwd}")
    print(f"  bwd x0 - x0     = {x0_bwd - x0}")
    print(f"  fwd xT          = {xT_fwd}")
    print(f"  fwd xT norm sq. = {jp.sum(xT_fwd**2)}")
    print(f"  elapsed         = {time.time() - t0}s")

  cost_T_fwd_per_iter = jp.array(cost_T_fwd_per_iter)
  xT_fwd_per_iter = jp.array(xT_fwd_per_iter)
  cost_0_bwd_per_iter = jp.array(cost_0_bwd_per_iter)
  x0_bwd_per_iter = jp.array(x0_bwd_per_iter)

  ### Plot performance per iteration, incl. average optimal policy performance.
  _, ax1 = plt.subplots()
  ax1.set_xlabel("Iteration")
  ax1.set_ylabel("Cost", color="tab:blue")
  ax1.set_yscale("log")
  ax1.tick_params(axis="y", labelcolor="tab:blue")
  ax1.plot(cost_T_fwd_per_iter, color="tab:blue")
  plt.axhline(opt_cost_fwd, linestyle="--", color="gray")

  ax2 = ax1.twinx()
  ax2.set_ylabel("Backward solve L2^2 error (cost in red, x(t) in purple, x(T) norm sq. in brown)",
                 color="tab:red")
  ax2.set_yscale("log")
  ax2.tick_params(axis="y", labelcolor="tab:red")
  ax2.plot(cost_0_bwd_per_iter**2, color="tab:red")
  ax2.plot(jp.sum((x0_bwd_per_iter - x0)**2, axis=-1), color="tab:purple")
  ax2.plot(jp.sum(xT_fwd_per_iter**2, axis=-1), color="tab:brown")

  plt.title(f"ODE control of LQR problem")

  blt.show()

if __name__ == "__main__":
  main()
