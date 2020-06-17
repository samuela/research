import time

import jax.numpy as jp
import matplotlib.pyplot as plt
from jax import random, vmap
from jax.experimental import ode, optimizers, stax
from jax.experimental.stax import Dense, Tanh

from research import blt
# from research.estop.pendulum.env import viz_pendulum_rollout
from research.odecontrol.pendulum import pendulum_dynamics
from research.odecontrol.radau_ode import policy_cost_and_grad
from research.utils import make_optimizer

def sample_x0(rng):
  rng_theta, rng_thetadot = random.split(rng)
  return jp.array([
      random.uniform(rng_theta, minval=0, maxval=2 * jp.pi),
      random.uniform(rng_thetadot, minval=-1, maxval=1)
  ])

def main():
  total_secs = 5.0
  num_iter = 10000
  rng = random.PRNGKey(0)

  dynamics = pendulum_dynamics(
      mass=0.1,
      length=1.0,
      gravity=9.8,
      friction=0.0,
  )

  def cost(x, u):
    assert x.shape == (2, )
    assert u.shape == (1, )
    # This is equivalent to OpenAI gym cost defined here: https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L51.
    theta = x[0] % (2 * jp.pi)
    return (theta - jp.pi)**2 + 0.1 * (x[1]**2) + 0.001 * (u[0]**2)

  policy_init, policy_nn = stax.serial(
      Dense(64),
      Tanh,
      Dense(1),
  )

  # Should it matter whether theta is wrapped into [0, 2pi]?
  policy = lambda params, x: policy_nn(
      params, jp.array([x[0] % (2 * jp.pi), x[1],
                        jp.cos(x[0]), jp.sin(x[0])]))

  rng_init_params, rng = random.split(rng)
  _, init_policy_params = policy_init(rng_init_params, (4, ))
  opt = make_optimizer(optimizers.adam(1e-3))(init_policy_params)
  loss_and_grad = policy_cost_and_grad(dynamics, cost, policy, example_x=jp.zeros(2))

  loss_per_iter = []
  elapsed_per_iter = []
  x0s = vmap(sample_x0)(random.split(rng, num_iter))
  for i in range(num_iter):
    t0 = time.time()
    loss, g = loss_and_grad(opt.value, x0s[i], total_secs)
    opt = opt.update(g)
    elapsed = time.time() - t0

    loss_per_iter.append(loss)
    elapsed_per_iter.append(elapsed)

    print(f"Episode {i}")
    print(f"    loss = {loss}")
    print(f"    elapsed = {elapsed}")

    # plot_control_contour(lambda x: policy(opt.value, x))
    # plt.title(f"Policy controls (episode = {(i + 1) * multi_steps})")

    # plot_policy_dynamics(dynamics, lambda x: policy(opt.value, x))
    # plt.title(f"Dynamics under the current policy (episode = {(i + 1) * multi_steps})")

    # plt.show()

  plt.figure()
  plt.plot(loss_per_iter)
  plt.title("ODE control of an inverted pendulum (linear policy)")
  plt.xlabel("Iteration")
  plt.ylabel(f"Policy cost (T = {total_secs}s)")
  # plt.savefig("ode_control_pendulum_linear.png")

  # viz
  framerate = 30
  # timesteps = jp.linspace(0, total_secs, num=int(total_secs * framerate))
  timesteps = jp.linspace(0, total_secs, num=int(10 * framerate))
  states = ode.odeint(lambda x, _: dynamics(x, policy(opt.value, x)), y0=x0s[0], t=timesteps)
  controls = vmap(lambda x: policy(opt.value, x))(states)

  plt.figure()
  plt.plot(states[:, 0], states[:, 1], marker='.')
  plt.xlabel("theta")
  plt.ylabel("theta dot")
  plt.title("Phase space trajectory")

  plt.figure()
  plt.plot(timesteps, controls)
  plt.xlabel("time")
  plt.ylabel("control input")
  plt.title("Policy control over time")

  plot_control_contour(lambda x: policy(opt.value, x))
  plot_policy_dynamics(dynamics, lambda x: policy(opt.value, x))

  blt.show()

  # viz_pendulum_rollout(states, controls)

def plot_control_contour(policy):
  t0 = time.time()
  plt.figure()
  thetas = jp.linspace(0, 2 * jp.pi, num=100)
  theta_dots = jp.linspace(-5, 5, num=100)
  z = vmap(policy)(jp.array([[th, thdot] for th in thetas for thdot in theta_dots]))
  plt.contourf(thetas, theta_dots, jp.reshape(z, (len(thetas), len(theta_dots))))
  plt.colorbar()
  plt.xlabel("theta")
  plt.ylabel("theta dot")
  plt.title("Policy controls")
  print(f"[timing] Plotting control contours took {time.time() - t0}s")

def plot_policy_dynamics(dynamics, policy):
  t0 = time.time()
  plt.figure()
  thetas = jp.linspace(0, 2 * jp.pi, num=100)
  theta_dots = jp.linspace(-10, 10, num=100)
  uv = vmap(lambda x: dynamics(x, policy(x)))(jp.array([[th, thdot] for th in thetas
                                                        for thdot in theta_dots]))
  uv_grid = jp.reshape(uv, (len(thetas), len(theta_dots), 2))
  plt.streamplot(thetas, theta_dots, uv_grid[:, :, 0], uv_grid[:, :, 1])
  plt.xlabel("theta")
  plt.ylabel("theta dot")
  plt.title("Dynamics under policy")
  print(f"[timing] Plotting control dynamics took {time.time() - t0}s")

if __name__ == "__main__":
  main()
