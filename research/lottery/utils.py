import jax.numpy as jnp
from jax import random

class RngPooper:
  """A stateful wrapper around stateless random.PRNGKey's."""
  def __init__(self, init_rng):
    self.rng = init_rng

  def poop(self):
    self.rng, rng_key = random.split(self.rng)
    return rng_key

def l1prox(x, alpha):
  return jnp.sign(x) * jnp.maximum(0, jnp.abs(x) - alpha)

def ec2_get_instance_type():
  # See also https://stackoverflow.com/questions/51486405/aws-ec2-command-line-display-instance-type/51486782
  return open("/sys/devices/virtual/dmi/id/product_name").read().strip()
