import time
import matplotlib.pyplot as plt
from jax import random, jit, lax, vmap
from jax.nn import initializers
import jax.numpy as jp
from jax.experimental import stax
from jax.experimental import ode
from jax.experimental import optimizers
from jax.experimental.stax import Dense, Tanh
from research.odecontrol.pendulum import pendulum_dynamics
from research.utils import make_optimizer
from research.estop.pendulum.env import viz_pendulum_rollout

def policy_cost_and_grad(dynamics, cost, policy, gamma=1.0):
  def ofunc(y, t, policy_params):
    x = y[1:]
    u = policy(policy_params, x)
    return jp.concatenate((jp.expand_dims((gamma**t) * cost(x, u), axis=0), dynamics(x, u)))

  def value_and_grad(policy_params, x0, total_time):
    y0 = jp.concatenate((jp.zeros((1, )), x0))

    # Zero is necessary for some reason...
    t = jp.array([0.0, total_time])

    primals, vjp = ode.vjp_odeint(ofunc, y0, t, policy_params)
    _, _, g = vjp(jp.expand_dims(jp.concatenate((jp.ones((1, )), jp.zeros_like(x0))), axis=0))
    return primals[1, 0], g

  return value_and_grad

def main():
  total_secs = 3.0
  rng = random.PRNGKey(0)

  dynamics = pendulum_dynamics(
      mass=0.1,
      length=1.0,
      gravity=9.8,
      friction=0.1,
  )

  def cost(x, u):
    assert x.shape == (2, )
    assert u.shape == (1, )
    theta = x[0] % (2 * jp.pi)
    return (theta - jp.pi)**2 + 0.1 * (x[1]**2) + 0.001 * (u[0]**2)

  policy_init, policy_nn = stax.serial(
      Dense(64),
      Tanh,
      Dense(64),
      Tanh,
      Dense(1, W_init=initializers.normal(stddev=1e-3), b_init=initializers.normal(stddev=1e-3)),
  )
  # policy_init, policy_nn = Dense(1,
  #                                W_init=initializers.normal(stddev=1e-3),
  #                                b_init=initializers.normal(stddev=1e-3))
  # Should it matter whether theta is wrapped into [0, 2pi]?
  policy = lambda params, x: policy_nn(
      params, jp.array([x[0] % (2 * jp.pi), x[1],
                        jp.cos(x[0]), jp.sin(x[0])]))

  cost_and_grad = jit(policy_cost_and_grad(dynamics, cost, policy, gamma=0.9))
  _, init_policy_params = policy_init(rng, (4, ))
  opt = make_optimizer(optimizers.adam(1e-3))(init_policy_params)

  def multiple_steps(num_steps):
    def body(_, stuff):
      _, opt = stuff
      cost, g = cost_and_grad(opt.value, jp.array([jp.pi, 0.01]), total_secs)
      return cost, opt.update(g)

    return lambda opt: lax.fori_loop(0, num_steps, body, (jp.zeros(()), opt))

  multi_steps = 1000
  run = jit(multiple_steps(multi_steps))

  costs = []
  for i in range(10):
    t0 = time.time()
    cost, opt = run(opt)
    # cost, g = cost_and_grad(opt.value, jp.array([jp.pi, 0.01]), total_secs)
    # opt = opt.update(g)
    print(f"Episode {(i + 1) * multi_steps}: cost = {cost}, elapsed = {time.time() - t0}")
    costs.append(float(cost))

    # plot_control_contour(lambda x: policy(opt.value, x))
    # plt.title(f"Policy controls (episode = {(i + 1) * multi_steps})")

    # plot_policy_dynamics(dynamics, lambda x: policy(opt.value, x))
    # plt.title(f"Dynamics under the current policy (episode = {(i + 1) * multi_steps})")

    # plt.show()

  plt.figure()
  plt.plot(costs)
  plt.title("ODE control of an inverted pendulum (linear policy)")
  plt.xlabel("Iteration")
  plt.ylabel(f"Policy cost (T = {total_secs}s)")
  # plt.savefig("ode_control_pendulum_linear.png")

  # viz
  framerate = 30
  # timesteps = jp.linspace(0, total_secs, num=int(total_secs * framerate))
  timesteps = jp.linspace(0, 10, num=int(10 * framerate))
  states = ode.odeint(
      lambda x, _: dynamics(x, policy(opt.value, x)),
      # y0=jp.zeros((2, )),
      y0=jp.array([jp.pi, 0.001]),
      t=timesteps)
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
  # plot_policy_dynamics(dynamics, lambda x: policy(opt.value, x))

  plt.show()

  viz_pendulum_rollout(states, controls)

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
