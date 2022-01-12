"""Koopman learning for the pendulum.

No control in this problem. We just try to learn the koopman dynamics and a
decoder to make sure that we learn a useful set of encoder functions.

TODO:
- log separate losses
- scatter plot of the eigenvalues of A
- phase space quiver plot
- The SIREN paper suggests that sin(x) is a good activation function.
- What's the right value for lam?
- Do we need to be comparing angles mod 2pi? Doesn't seem to be an issue for training?
"""

import time

import jax.numpy as jnp
import optax
from flax import linen as nn
from jax import grad, jit, jvp, lax, random, value_and_grad, vmap
from jax.experimental.ode import odeint
from jax.lax import stop_gradient
from jax.scipy.linalg import expm
from tqdm import tqdm

import wandb

class RngPooper:
  """A stateful wrapper around stateless random.PRNGKey's."""
  def __init__(self, init_rng):
    self.rng = init_rng

  def poop(self):
    self.rng, rng_key = random.split(self.rng)
    return rng_key

def ec2_get_instance_type():
  # See also https://stackoverflow.com/questions/51486405/aws-ec2-command-line-display-instance-type/51486782
  return open("/sys/devices/virtual/dmi/id/product_name").read().strip()

# See https://twitter.com/jon_barron/status/1387167648669048833
def squareplus(x):
  return lax.mul(0.5, lax.add(x, lax.sqrt(lax.add(lax.square(x), 4.0))))

def circle_dist(a, b):
  return jnp.minimum(
      jnp.abs(a - b),
      jnp.minimum(
          jnp.abs(jnp.pi - b) + jnp.abs(-jnp.pi - a),
          jnp.abs(jnp.pi - a) + jnp.abs(-jnp.pi - b)))

if __name__ == "__main__":
  # , mode="disabled"
  wandb.init(project="koopy-poopy", entity="skainswo")

  config = wandb.config
  config.ec2_instance_type = ec2_get_instance_type()

  # Optimization parameters:
  config.learning_rate = 0.001
  config.batch_size = 128
  config.num_iterations = 1000

  # Model parameters:
  config.dim_z = 128
  # The weight of the autoencoder loss term.
  config.lam = 1
  config.gravity = 9.8
  config.length = 1

  dim_x = 2

  rp = RngPooper(random.PRNGKey(0))

  def dynamics(x):
    # Returns dx/dt = [th_dot, th_ddot]
    # theta = 0.0 (or 2pi) corresponds to the pendulum pointing straight down.
    th, th_dot = x
    th_ddot = -config.gravity / config.length * jnp.sin(th)
    return jnp.array([th_dot, th_ddot])

  def sample_random_x(rng):
    rng1, rng2 = random.split(rng)
    return jnp.array([
        random.uniform(rng1, minval=-jnp.pi, maxval=jnp.pi),
        random.uniform(rng2, minval=-10, maxval=10)
    ])

  sim_timesteps = jnp.linspace(0, 10, num=100)

  def ground_truth_sim(x0):
    return odeint(lambda x, t: dynamics(x), x0, sim_timesteps)

  def koopman_sim(params, x0):
    # x(t) = decode(exp(A * t) encode(x0))
    A, encoder_params, decoder_params = params
    z0 = encoder.apply(encoder_params, jnp.expand_dims(x0, 0))[0, :]
    return decoder.apply(decoder_params, vmap(lambda t: expm(t * A) @ z0)(sim_timesteps))

  def rem2pi(x):
    """Wrap x to [-pi, pi]"""
    # TODO do we really need stop_gradient? Does jnp.round stop gradients?
    return x - stop_gradient(2 * jnp.pi * jnp.round(x / (2 * jnp.pi)))

  activation = squareplus

  class Encoder(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = jnp.hstack([x, jnp.cos(x), jnp.sin(x)])
      x = activation(nn.Dense(128)(x))
      x = activation(nn.Dense(128)(x))
      x = nn.Dense(config.dim_z)(x)
      return x

  class Decoder(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = activation(nn.Dense(128)(x))
      x = activation(nn.Dense(128)(x))
      x = nn.Dense(dim_x)(x)
      th, th_dot = x[:, 0], x[:, 1]
      return jnp.array([rem2pi(th), th_dot])

  encoder = Encoder()
  decoder = Decoder()

  def koopman_loss_one(params, x):
    A, encoder_params, _ = params
    # b = \nabla_x g(x) * F(x)
    Fx = dynamics(x)
    g = lambda xx: encoder.apply(encoder_params, jnp.expand_dims(xx, 0))[0, :]
    z, b = jvp(g, (x, ), (Fx, ))
    return jnp.sum((A @ z - b)**2) / config.dim_z

  def autoencoder_loss_one(params, x):
    _, encoder_params, decoder_params = params
    th, th_dot = x
    z = encoder.apply(encoder_params, jnp.expand_dims(x, 0))
    pred_th, pred_th_dot = decoder.apply(decoder_params, z)
    return (circle_dist(th, pred_th)**2 + (th_dot - pred_th_dot)**2) / dim_x

  def loss_one(params, x):
    return koopman_loss_one(params, x) + config.lam * autoencoder_loss_one(params, x)

  def loss(params, xs_batch):
    return jnp.mean(vmap(loss_one, in_axes=(None, 0))(params, xs_batch))

  tx = optax.adam(config.learning_rate)

  @jit
  def update(carry, rng):
    # TODO log the two losses separately?
    opt_state, params = carry
    xs_batch = vmap(sample_random_x)(random.split(rng, config.batch_size))
    batch_loss, g = value_and_grad(loss)(params, xs_batch)
    updates, opt_state = tx.update(g, opt_state)
    params = optax.apply_updates(params, updates)
    return (opt_state, params), batch_loss

  # TODO can we get rid of the 1 in the shape when initializing?
  encoder_params = encoder.init(rp.poop(), jnp.zeros((1, dim_x)))
  decoder_params = decoder.init(rp.poop(), jnp.zeros((1, config.dim_z)))
  A = 1e-3 * random.normal(rp.poop(), shape=(config.dim_z, config.dim_z))
  params = (A, encoder_params, decoder_params)

  # This is the initial state for the debugging visualization during training.
  example_x0 = jnp.array([1.5, 0.0])

  # TODO: do many updates in a jax.fori_loop...
  # For some reason we must pass `params` to `tx.init` but `opt_state` does not
  # contain `params`. This is just how optax is designed apparently.
  opt_state = tx.init(params)
  start_time = time.time()
  for step in tqdm(range(config.num_iterations)):
    (opt_state, params), losses = lax.scan(update, (opt_state, params),
                                           random.split(rp.poop(), 1000))
    wandb.log({
        "mean_loss":
        jnp.mean(losses),
        "losses":
        losses,
        "step":
        step,
        "example_rollout":
        wandb.plot.line_series(
            xs=sim_timesteps,
            ys=[ground_truth_sim(example_x0)[:, 0],
                koopman_sim(params, example_x0)[:, 0]],
            keys=["ground truth", "koopman"],
            title="Example rollout",
            xname="Time (s)"),
        "wallclock":
        time.time() - start_time
    })
