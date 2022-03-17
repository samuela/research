import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import linen as nn
from jax import random, tree_map

from utils import unflatten_params

rng = random.PRNGKey(0)

class Model(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(2)(x)
    x = nn.relu(x)
    x = nn.Dense(2)(x)
    x = nn.relu(x)
    x = nn.Dense(1)(x)
    return x

model = Model()

# dense kernel shape: (in, out)
dtype = jnp.float32
paramsA = {
    "Dense_0/kernel": jnp.array([[1, 0], [0, -1]], dtype=dtype),
    "Dense_0/bias": jnp.array([1, 0], dtype=dtype),
    "Dense_1/kernel": jnp.array([[-1, 0], [0, 1]], dtype=dtype),
    "Dense_1/bias": jnp.array([1, 0], dtype=dtype),
    "Dense_2/kernel": jnp.array([[-1], [-1]], dtype=dtype),
    "Dense_2/bias": jnp.array([0], dtype=dtype),
}
paramsB1 = {
    "Dense_0/kernel": jnp.array([[-1, 0], [0, 1]], dtype=dtype),
    "Dense_0/bias": jnp.array([0, 1], dtype=dtype),
    "Dense_1/kernel": jnp.array([[1, 0], [0, -1]], dtype=dtype),
    "Dense_1/bias": jnp.array([0, 1], dtype=dtype),
    "Dense_2/kernel": jnp.array([[-1], [-1]], dtype=dtype),
    "Dense_2/bias": jnp.array([0], dtype=dtype),
}

def swap_layer(layer: int, params):
  ix = jnp.array([1, 0])
  return {
      **params,
      f"Dense_{layer}/kernel": params[f"Dense_{layer}/kernel"][:, ix],
      f"Dense_{layer}/bias": params[f"Dense_{layer}/bias"][ix],
      f"Dense_{layer+1}/kernel": params[f"Dense_{layer+1}/kernel"][ix, :],
  }

swap_first_layer = lambda params: swap_layer(0, params)
swap_second_layer = lambda params: swap_layer(1, params)

paramsB2 = swap_first_layer(paramsB1)
paramsB3 = swap_second_layer(paramsB1)
paramsB4 = swap_first_layer(swap_second_layer(paramsB1))

num_examples = 1024
testX = random.uniform(rng, (num_examples, 2), dtype=dtype, minval=-1, maxval=1)
testY = (testX[:, 0] >= 0) & (testX[:, 1] >= 0)

def accuracy(params):
  return jnp.sum((model.apply({"params": unflatten_params(params)}, testX) >= 0
                  ).flatten() == testY) / num_examples

assert accuracy(paramsA) == 1.0
assert accuracy(paramsB1) == 1.0
assert accuracy(paramsB2) == 1.0
assert accuracy(paramsB3) == 1.0
assert accuracy(paramsB4) == 1.0

def interp_params(lam, pA, pB):
  return tree_map(lambda a, b: lam * a + (1 - lam) * b, pA, pB)

lambdas = jnp.linspace(0, 1, num=128)
interp1 = jnp.array([accuracy(interp_params(lam, paramsA, paramsB1)) for lam in lambdas])
interp2 = jnp.array([accuracy(interp_params(lam, paramsA, paramsB2)) for lam in lambdas])
interp3 = jnp.array([accuracy(interp_params(lam, paramsA, paramsB3)) for lam in lambdas])
interp4 = jnp.array([accuracy(interp_params(lam, paramsA, paramsB4)) for lam in lambdas])

def plot_interp_loss():
  fig = plt.figure()
  ax = fig.add_subplot(111)
  # We make losses start at 0, since that intuitively makes more sense.
  ax.plot(lambdas, -interp1 + 1, linewidth=2)
  ax.plot(lambdas, -interp2 + 1, linewidth=2)
  ax.plot(lambdas, -interp3 + 1, linewidth=2)
  ax.plot(lambdas, -interp4 + 1, linewidth=2)
  ax.plot([-1, 2], [0, 0],
          linestyle="dashed",
          color="tab:grey",
          alpha=0.5,
          label="Optimal performance")
  ax.set_xlabel("$\lambda$")
  ax.set_xticks([0, 1])
  ax.set_xticklabels(["Model $A$", "Model $B$"])
  ax.set_xlim(-0.05, 1.05)
  ax.set_ylabel("Loss")
  ax.set_title("No permutation yields linear mode connectivity")
  ax.legend(framealpha=0.5)
  fig.tight_layout()
  return fig

fig = plot_interp_loss()
plt.savefig(f"sgd_is_special_loss_interp.png", dpi=300)
plt.savefig(f"sgd_is_special_loss_interp.pdf")
plt.close(fig)
