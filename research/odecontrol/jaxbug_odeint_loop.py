from jax import random, jit, lax, vmap
from jax.nn import initializers
import jax.numpy as jp
from jax.experimental import stax
from jax.experimental import ode
from jax.experimental import optimizers
from jax.experimental.stax import Dense
from research.odecontrol.pendulum import pendulum_dynamics

def policy_cost_and_grad(dynamics, cost, policy):
  def ofunc(y, _t, policy_params):
    x = y[1:]
    u = policy(policy_params, x)
    return jp.concatenate((jp.expand_dims(cost(x, u), axis=0), dynamics(x, u)))

  def value_and_grad(policy_params, x0, total_time):
    y0 = jp.concatenate((jp.zeros((1, )), x0))

    # Zero is necessary for some reason...
    t = jp.array([0.0, total_time])

    primals, vjp = ode.vjp_odeint(ofunc, y0, t, policy_params)
    _, _, g = vjp(jp.expand_dims(jp.concatenate((jp.ones((1, )), jp.zeros_like(x0))), axis=0))
    return primals[1, 0], g

  return value_and_grad

def main():
  total_secs = 120.0
  rng = random.PRNGKey(0)

  dynamics = pendulum_dynamics(
      mass=0.1,
      length=1.0,
      gravity=9.8,
      friction=0.1,
  )

  def cost(x, u):
    theta = x[0] % (2 * jp.pi)
    return (theta - jp.pi)**2 + 0.1 * (x[1]**2) + 0.001 * (u[0]**2)

  policy_init, policy_nn = Dense(1)
  policy = lambda params, x: policy_nn(params, jp.array([x[0], x[1], jp.cos(x[0]), jp.sin(x[0])]))

  cost_and_grad = jit(policy_cost_and_grad(dynamics, cost, policy))
  _, init_policy_params = policy_init(rng, (4, ))

  print("about to get stuck forever...")
  cost, g = cost_and_grad(init_policy_params, jp.array([jp.pi, 0.01]), total_secs)
  print("this will never happen")

if __name__ == "__main__":
  main()
