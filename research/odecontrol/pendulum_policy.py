import time

import jax.numpy as jnp
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
  return jnp.array([
      random.uniform(rng_theta, minval=0, maxval=2 * jnp.pi),
      random.uniform(rng_thetadot, minval=-1, maxval=1)
  ])

def main():
  num_iter = 10000
  # Most people run 1000 steps and the OpenAI gym pendulum is 0.05s per step.
  # The max torque that can be applied is also 2 in their setup.
  total_secs = 5.0
  max_torque = 2.0
  rng = random.PRNGKey(0)

  dynamics = pendulum_dynamics(
      mass=1.0,
      length=1.0,
      gravity=9.8,
      friction=0.0,
  )

  def cost(x, u):
    assert x.shape == (2, )
    assert u.shape == (1, )
    # This is equivalent to OpenAI gym cost defined here: https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L51.
    theta = x[0] % (2 * jnp.pi)
    return (theta - jnp.pi)**2 + 0.1 * (x[1]**2) + 0.001 * (u[0]**2)

  policy_init, policy_nn = stax.serial(
      Dense(64),
      Tanh,
      Dense(64),
      Tanh,
      Dense(1),
      Tanh,
      stax.elementwise(lambda x: max_torque * x),
  )

  # Should it matter whether theta is wrapped into [0, 2pi]?
  policy = lambda params, x: policy_nn(
      params, jnp.array([x[0] % (2 * jnp.pi), x[1],
                         jnp.cos(x[0]), jnp.sin(x[0])]))

  rng_init_params, rng = random.split(rng)
  _, init_policy_params = policy_init(rng_init_params, (4, ))
  opt = make_optimizer(optimizers.adam(1e-3))(init_policy_params)
  loss_and_grad = policy_cost_and_grad(dynamics, cost, policy, example_x=jnp.zeros(2))

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

  blt.remember({
      "loss_per_iter": loss_per_iter,
      "elapsed_per_iter": elapsed_per_iter,
      "final_params": opt.value
  })

  plt.figure()
  plt.plot(loss_per_iter)
  plt.yscale("log")
  plt.title("ODE control of an inverted pendulum")
  plt.xlabel("Iteration")
  plt.ylabel(f"Policy cost (T = {total_secs}s)")

  # Viz
  num_viz_rollouts = 50
  framerate = 30
  timesteps = jnp.linspace(0, total_secs, num=int(total_secs * framerate))
  rollout = lambda x0: ode.odeint(
      lambda x, _: dynamics(x, policy(opt.value, x)), y0=x0, t=timesteps)

  plt.figure()
  states = rollout(jnp.zeros(2))
  plt.plot(states[:, 0], states[:, 1], marker=".")
  plt.xlabel("theta")
  plt.ylabel("theta dot")
  plt.title("Swing up trajectory")

  plt.figure()
  states = vmap(rollout)(x0s[:num_viz_rollouts])
  for i in range(num_viz_rollouts):
    plt.plot(states[i, :, 0], states[i, :, 1], marker='.', alpha=0.5)
  plt.xlabel("theta")
  plt.ylabel("theta dot")
  plt.title("Phase space trajectory")

  plot_control_contour(lambda x: policy(opt.value, x))
  plot_policy_dynamics(dynamics, lambda x: policy(opt.value, x))

  blt.show()

  # viz_pendulum_rollout(states, controls)

def plot_control_contour(policy):
  t0 = time.time()
  plt.figure()
  thetas = jnp.linspace(0, 2 * jnp.pi, num=100)
  theta_dots = jnp.linspace(-5, 5, num=100)
  z = vmap(policy)(jnp.array([[th, thdot] for th in thetas for thdot in theta_dots]))
  plt.contourf(thetas, theta_dots, jnp.reshape(z, (len(thetas), len(theta_dots))))
  plt.colorbar()
  plt.xlabel("theta")
  plt.ylabel("theta dot")
  plt.title("Policy controls")
  print(f"[timing] Plotting control contours took {time.time() - t0}s")

def plot_policy_dynamics(dynamics, policy):
  t0 = time.time()
  thetas = jnp.linspace(0, 2 * jnp.pi, num=100)
  theta_dots = jnp.linspace(-10, 10, num=100)

  dyn = lambda x: dynamics(x, policy(x))
  uv = vmap(lambda th: vmap(lambda thdot: dyn(jnp.array([th, thdot])))(theta_dots))(thetas)
  uv = jnp.transpose(uv, (1, 0, 2))

  # Quiver plot.
  plt.figure()
  plt.quiver(thetas, theta_dots, uv[:, :, 0], uv[:, :, 1])
  plt.xlabel("theta")
  plt.ylabel("theta dot")
  plt.title("Dynamics under policy")

  # Stream plot is way slower.
  plt.figure()
  plt.streamplot(thetas, theta_dots, uv[:, :, 0], uv[:, :, 1])
  plt.xlabel("theta")
  plt.ylabel("theta dot")
  plt.title("Dynamics under policy")

  print(f"[timing] Plotting control dynamics took {time.time() - t0}s")

if __name__ == "__main__":
  main()
