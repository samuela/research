from jax import random

class RngPooper:
  """A stateful wrapper around stateless random.PRNGKey's."""
  def __init__(self, init_rng):
    self.rng = init_rng

  def poop(self):
    self.rng, rng_key = random.split(self.rng)
    return rng_key
