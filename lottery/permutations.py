"""Permuting neural networks to look like one another."""
import jax.numpy as jnp
from flax import linen as nn
from jax import random
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from utils import RngPooper, flatten_params, kmatch, unflatten_params

def cosine_similarity(X, Y):
  # X: (m, d)
  # Y: (n, d)
  # return: (m, n)
  return (X @ Y.T) / jnp.linalg.norm(X, axis=-1).reshape((-1, 1)) / jnp.linalg.norm(Y, axis=-1)

def permutify(params1, params2):
  """Permute the parameters of params2 to match params1 as closely as possible.
  Returns the permuted version of params2. Only works on sequences of Dense
  layers for now."""
  p1f = flatten_params(params1)
  p2f = flatten_params(params2)

  p2f_new = {**p2f}
  num_layers = max(int(kmatch("params/Dense_*/**", k).group(1)) for k in p1f.keys())
  # range is [0, num_layers), so we're safe here since we don't want to be
  # reordering the output of the last layer.
  for layer in range(num_layers):
    # Maximize since we're dealing with similarities, not distances.
    ri, ci = linear_sum_assignment(cosine_similarity(p1f[f"params/Dense_{layer}/kernel"].T,
                                                     p2f_new[f"params/Dense_{layer}/kernel"].T),
                                   maximize=True)
    assert (ri == jnp.arange(len(ri))).all()

    p2f_new = {
        **p2f_new, f"params/Dense_{layer}/kernel": p2f_new[f"params/Dense_{layer}/kernel"][:, ci],
        f"params/Dense_{layer}/bias": p2f_new[f"params/Dense_{layer}/bias"][ci],
        f"params/Dense_{layer+1}/kernel": p2f_new[f"params/Dense_{layer+1}/kernel"][ci, :]
    }

  new_params2 = unflatten_params(p2f_new)

  return new_params2

if __name__ == "__main__":
  rp = RngPooper(random.PRNGKey(0))

  print("Testing cosine_similarity...")
  for _ in range(10):
    X = random.normal(rp.poop(), (3, 5))
    Y = random.normal(rp.poop(), (7, 5))
    assert jnp.allclose(1 - cosine_similarity(X, Y), cdist(X, Y, metric="cosine"))

  print("Testing permutify...")

  class Model(nn.Module):

    @nn.compact
    def __call__(self, x):
      x = nn.Dense(1024, bias_init=nn.initializers.normal(stddev=1.0))(x)
      x = nn.relu(x)
      x = nn.Dense(1024, bias_init=nn.initializers.normal(stddev=1.0))(x)
      x = nn.relu(x)
      x = nn.Dense(10)(x)
      x = nn.log_softmax(x)
      return x

  model = Model()
  p1 = model.init(rp.poop(), jnp.zeros((1, 28 * 28)))
  p2 = model.init(rp.poop(), jnp.zeros((1, 28 * 28)))
  # print(tree_map(jnp.shape, flatten_params(p1)))

  new_p2 = permutify(p1, p2)

  # Test that the model is the same after permutation.
  random_input = random.normal(rp.poop(), (128, 28 * 28))
  # print(jnp.max(jnp.abs(model.apply(p2, random_input) - model.apply(new_p2, random_input))))
  assert ((jnp.abs(model.apply(p2, random_input) - model.apply(new_p2, random_input))) < 1e-5).all()
