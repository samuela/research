import time
import control
import matplotlib.pyplot as plt
from jax import random
from jax import jit
from jax import value_and_grad
from jax import lax
from jax import vmap
import jax.numpy as jp
from jax.experimental import stax
from jax.experimental import ode
from jax.experimental import optimizers
from jax.experimental.stax import Dense
from jax.experimental.stax import Relu
from research.utils import make_optimizer

def policy_integrate_cost(dynamics_fn, cost_fn, policy, gamma):
  def ofunc(y, t, policy_params):
    x = y[1:]
    u = policy(policy_params, x)
    return jp.concatenate((jp.expand_dims((gamma**t) * cost_fn(x, u), axis=0), dynamics_fn(x, u)))

  def evally(policy_params, x0, total_time):
    # Zero is necessary for some reason...
    t = jp.array([0.0, total_time])
    y0 = jp.concatenate((jp.zeros((1, )), x0))
    yT = ode.odeint(ofunc, y0, t, policy_params, rtol=1e-3, mxstep=1e6)
    return yT[1, 0]

  return evally

def main():
  total_secs = 10.0
  gamma = 0.9
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

  # rngA, rngB, rngQ, rngR, rng = random.split(rng, 5)
  # # A = random.normal(rngA, (2, 2))
  # A = -1 * random_psd(rngA, 2)
  # B = random.normal(rngB, (2, 2))
  # Q = random_psd(rngQ, 2) + 0.1 * jp.eye(2)
  # R = random_psd(rngR, 2) + 0.1 * jp.eye(2)
  # N = jp.zeros((2, 2))

  # x_dim, u_dim = B.shape

  dynamics_fn = lambda x, u: A @ x + B @ u
  cost_fn = lambda x, u: x.T @ Q @ x + u.T @ R @ u + 2 * x.T @ N @ u

  ### Solve the Riccatti equation to get the infinite-horizon optimal solution.
  K, _, _ = control.lqr(A, B, Q, R, N)
  K = jp.array(K)

  t0 = time.time()
  rng_eval, rng = random.split(rng)
  x0_eval = random.normal(rng_eval, (1000, 2))
  opt_all_costs = vmap(lambda x0: policy_integrate_cost(dynamics_fn, cost_fn, lambda _, x: -K @ x,
                                                        gamma)(None, x0, total_secs))(x0_eval)
  opt_cost = jp.mean(opt_all_costs)
  print(f"opt_cost = {opt_cost} in {time.time() - t0}s")

  ### Set up the learned policy model.
  policy_init, policy = stax.serial(
      Dense(64),
      Relu,
      Dense(64),
      Relu,
      Dense(2),
  )
  # policy_init, policy = DenseNoBias(2)

  rng_init_params, rng = random.split(rng)
  _, init_policy_params = policy_init(rng_init_params, (2, ))

  cost_and_grad = jit(value_and_grad(policy_integrate_cost(dynamics_fn, cost_fn, policy, gamma)))
  opt = make_optimizer(optimizers.adam(1e-3))(init_policy_params)

  def multiple_steps(num_steps):
    """Return a jit-able function that runs `num_steps` iterations."""
    def body(_, stuff):
      rng, _, opt = stuff
      rng_x0, rng = random.split(rng)
      x0 = random.normal(rng_x0, (2, ))
      cost, g = cost_and_grad(opt.value, x0, total_secs)

      # Gradient clipping
      # g = tree_map(lambda x: jp.clip(x, -10, 10), g)
      # g = optimizers.clip_grads(g, 64)

      return rng, cost, opt.update(g)

    return lambda rng, opt: lax.fori_loop(0, num_steps, body, (rng, jp.zeros(()), opt))

  multi_steps = 1
  run = jit(multiple_steps(multi_steps))

  ### Main optimization loop.
  costs = []
  for i in range(25000):
    t0 = time.time()
    rng, cost, opt = run(rng, opt)
    print(
        f"Episode {(i + 1) * multi_steps}: excess cost = {cost - opt_cost}, elapsed = {time.time() - t0}"
    )
    costs.append(float(cost))

  print(f"Opt solution cost from starting point: {opt_cost}")
  # print(f"Gradient at opt solution: {opt_g}")

  # Print the identified and optimal policy. Note that layers multiply multipy
  # on the right instead of the left so we need a transpose.
  print(f"Est solution parameters: {opt.value}")
  print(f"Opt solution parameters: {-K.T}")

  est_all_costs = vmap(lambda x0: policy_integrate_cost(dynamics_fn, cost_fn, policy, gamma)
                       (opt.value, x0, total_secs))(x0_eval)

  ### Scatter plot of learned policy performance vs optimal policy performance.
  plt.figure()
  plt.scatter(est_all_costs, opt_all_costs)
  plt.plot([-100, 100], [-100, 100], color="gray")
  plt.xlim(0, jp.max(est_all_costs))
  plt.ylim(0, jp.max(opt_all_costs))
  plt.xlabel("Learned policy cost")
  plt.ylabel("Optimal cost")
  plt.title("Performance relative to the direct LQR solution")

  ### Plot performance per iteration, incl. average optimal policy performance.
  plt.figure()
  plt.plot(costs)
  plt.axhline(opt_cost, linestyle="--", color="gray")
  plt.yscale("log")
  plt.xlabel("Iteration")
  plt.ylabel(f"Cost (T = {total_secs}s)")
  plt.legend(["Learned policy", "Direct LQR solution"])
  plt.title("ODE control of LQR problem")

  ### Example rollout plots (learned policy vs optimal policy).
  x0 = jp.array([1.0, 2.0])
  framerate = 30
  timesteps = jp.linspace(0, total_secs, num=int(total_secs * framerate))
  est_policy_rollout_states = ode.odeint(lambda x, _: dynamics_fn(x, policy(opt.value, x)),
                                         y0=x0,
                                         t=timesteps)
  est_policy_rollout_controls = vmap(lambda x: policy(opt.value, x))(est_policy_rollout_states)

  opt_policy_rollout_states = ode.odeint(lambda x, _: dynamics_fn(x, -K @ x), y0=x0, t=timesteps)
  opt_policy_rollout_controls = vmap(lambda x: -K @ x)(opt_policy_rollout_states)

  plt.figure()
  plt.plot(est_policy_rollout_states[:, 0], est_policy_rollout_states[:, 1], marker='.')
  plt.plot(opt_policy_rollout_states[:, 0], opt_policy_rollout_states[:, 1], marker='.')
  plt.xlabel("x_1")
  plt.ylabel("x_2")
  plt.legend(["Learned policy", "Direct LQR solution"])
  plt.title("Phase space trajectory")

  plt.figure()
  plt.plot(timesteps, jp.sqrt(jp.sum(est_policy_rollout_controls**2, axis=-1)))
  plt.plot(timesteps, jp.sqrt(jp.sum(opt_policy_rollout_controls**2, axis=-1)))
  plt.xlabel("time")
  plt.ylabel("control input (L2 norm)")
  plt.legend(["Learned policy", "Direct LQR solution"])
  plt.title("Policy control over time")

  ### Plot quiver field showing dynamics under learned policy.
  plot_policy_dynamics(dynamics_fn, cost_fn, lambda x: policy(opt.value, x))

  plt.show()

def plot_policy_dynamics(dynamics_fn, cost_fn, policy):
  t0 = time.time()
  plt.figure()

  x1s = jp.linspace(-1, 1, num=50)
  x2s = jp.linspace(-1, 1, num=50)
  flatmesh = jp.array([[x1, x2] for x1 in x1s for x2 in x2s])
  uv = vmap(lambda x: dynamics_fn(x, policy(x)))(flatmesh)
  uv_grid = jp.reshape(uv, (len(x1s), len(x2s), 2))
  color = vmap(lambda x: cost_fn(x, policy(x)))(flatmesh)
  color_grid = jp.reshape(color, (len(x1s), len(x2s)))

  plt.quiver(x1s, x2s, uv_grid[:, :, 0], uv_grid[:, :, 1], color_grid)
  plt.axis("equal")
  plt.xlabel("x_1")
  plt.ylabel("x_2")
  plt.title("Dynamics under policy")
  print(f"[timing] Plotting control dynamics took {time.time() - t0}s")

if __name__ == "__main__":
  main()
