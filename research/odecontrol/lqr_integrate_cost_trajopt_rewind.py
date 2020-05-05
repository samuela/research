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
from jax.experimental.stax import Relu
from jax.experimental.stax import Tanh
from research.utils import make_optimizer
from research.utils import DenseNoBias
from research import blt

def fixed_env(n):
  A = 0 * jp.eye(n)
  # A = jp.diag(jp.array([-1.0, 1.0]))
  B = jp.eye(n)
  Q = jp.eye(n)
  R = jp.eye(n)
  N = jp.zeros((n, n))
  return A, B, Q, R, N

def policy_integrate_cost(x_dim, dynamics_fn, cost_fn, gamma):
  # Specialize to the environment.

  enc_init, enc = stax.serial(
      Dense(256),
      Tanh,
      Dense(256),
      Tanh,
      Dense(x_dim),
  )
  _, enc_params = enc_init(random.PRNGKey(0), (2 * x_dim, ))

  def eval_policy(policy):
    # Specialize to the policy.

    def ofunc(y, t, policy_params):
      x = y[1:x_dim + 1]
      z = y[x_dim + 1:]
      u = policy(policy_params, x)
      cost_dt = jp.expand_dims((gamma**t) * cost_fn(x, u), axis=0)
      x_dt = dynamics_fn(x, u)
      z_dt = enc(enc_params, jp.concatenate((x, z)))
      return jp.concatenate((cost_dt, x_dt, z_dt))

    def eval_from_x0(policy_params, x0, total_time):
      # Zero is necessary for some reason...
      t = jp.array([0.0, total_time])
      y0 = jp.concatenate((jp.zeros((1, )), x0, jp.zeros_like(x0)))
      # odeint_kwargs = {"rtol": 1e-3, "mxstep": 1e6}
      odeint_kwargs = {"mxstep": 1e6}
      y_fwd = ode.odeint(ofunc, y0, t, policy_params, **odeint_kwargs)

      # This is similar but not exactly the same as the place that the rev-mode
      # solution since the step sizes can vary when using all the other
      # parameters.
      y_bwd = ode.odeint(lambda y, t, *args: -ofunc(y, -t, *args), y_fwd[1], -t[::-1],
                         policy_params, **odeint_kwargs)

      return y_fwd, y_bwd[::-1]

    return eval_from_x0

  return eval_policy

def main():
  total_time = 20.0
  gamma = 1.0
  x_dim = 2
  rng = random.PRNGKey(0)

  x0 = jp.array([2.0, 1.0])
  # rng_x0, rng = random.split(rng)
  # x0 = random.normal(rng_x0, shape=(x_dim, ))

  ### Set up the problem/environment
  # xdot = Ax + Bu
  # u = - Kx
  # cost = xQx + uRu + 2xNu
  A, B, Q, R, N = fixed_env(x_dim)
  dynamics_fn = lambda x, u: A @ x + B @ u
  cost_fn = lambda x, u: x.T @ Q @ x + u.T @ R @ u + 2 * x.T @ N @ u
  policy_loss = policy_integrate_cost(x_dim, dynamics_fn, cost_fn, gamma)

  ### Solve the Riccatti equation to get the infinite-horizon optimal solution.
  K, _, _ = control.lqr(A, B, Q, R, N)
  K = jp.array(K)

  t0 = time.time()
  opt_y_fwd, opt_y_bwd = policy_loss(lambda _, x: -K @ x)(None, x0, total_time)
  opt_cost = opt_y_fwd[1, 0]
  print(f"opt_cost = {opt_cost} in {time.time() - t0}s")
  print(opt_y_fwd)
  print(opt_y_bwd)
  print(f"l2 error: {jp.sqrt(jp.sum((opt_y_fwd - opt_y_bwd)**2))}")

  ### Set up the learned policy model.
  policy_init, policy = stax.serial(
      Dense(64),
      Tanh,
      # Dense(64),
      # Tanh,
      Dense(x_dim),
  )
  # policy_init, policy = DenseNoBias(2)

  rng_init_params, rng = random.split(rng)
  _, init_policy_params = policy_init(rng_init_params, (x_dim, ))
  opt = make_optimizer(optimizers.adam(1e-3))(init_policy_params)
  # cost_and_grad = jit(value_and_grad(policy_loss(policy)))
  runny_run = jit(policy_loss(policy))

  ### Main optimization loop.
  costs = []
  bwd_errors = []
  for i in range(5000):
    t0 = time.time()
    (y_fwd, y_bwd), vjp = jax.vjp(runny_run, opt.value, x0, total_time)
    cost = y_fwd[1, 0]

    y_fwd_bar = jax.ops.index_update(jp.zeros_like(y_fwd), (1, 0), 1)
    g, _, _ = vjp((y_fwd_bar, jp.zeros_like(y_bwd)))
    opt = opt.update(g)

    bwd_err = jp.sqrt(jp.sum((y_fwd - y_bwd)**2))
    bwd_errors.append(bwd_err)

    print(
        f"Episode {i}: excess cost = {cost - opt_cost}, bwd error = {bwd_err} elapsed = {time.time() - t0}"
    )
    costs.append(float(cost))

  print(f"Opt solution cost from starting point: {opt_cost}")
  # print(f"Gradient at opt solution: {opt_g}")

  # Print the identified and optimal policy. Note that layers multiply multipy
  # on the right instead of the left so we need a transpose.
  # print(f"Est solution parameters: {opt.value}")
  # print(f"Opt solution parameters: {-K.T}")

  ### Plot performance per iteration, incl. average optimal policy performance.
  _, ax1 = plt.subplots()
  ax1.set_xlabel("Iteration")
  ax1.set_ylabel("Cost", color="tab:blue")
  ax1.set_yscale("log")
  ax1.tick_params(axis="y", labelcolor="tab:blue")
  ax1.plot(costs, color="tab:blue")
  plt.axhline(opt_cost, linestyle="--", color="gray")

  ax2 = ax1.twinx()
  ax2.set_ylabel("Backward solve L2 error", color="tab:red")
  ax2.set_yscale("log")
  ax2.tick_params(axis="y", labelcolor="tab:red")
  ax2.plot(bwd_errors, color="tab:red")
  # plt.yscale("log")
  # plt.xlabel("Iteration")
  # plt.ylabel("Cost")
  # plt.legend(["Learned policy", "Direct LQR solution"])
  plt.title(f"ODE control of LQR problem")

  blt.show()

if __name__ == "__main__":
  main()
