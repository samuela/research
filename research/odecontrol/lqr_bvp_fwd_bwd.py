import time
import control
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from jax import vjp
from jax import vmap
from jax import jit
from jax import jacrev
from jax import random
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_map
from jax.tree_util import tree_multimap
from jax.experimental.stax import Dense
from jax.experimental.stax import Tanh
from jax.experimental import optimizers
from jax.experimental import stax
from research.utils import zeros_like_tree
from research.odecontrol.lqr_integrate_cost_trajopt_failure_case import fixed_env
from research.utils import make_optimizer
from research import blt

def bvp_fwd_bwd(f, y0, t0, t1, f_args, adj_y_t1, init_num_nodes=2):
  z_bc = (y0, adj_y_t1, 0.0, zeros_like_tree(f_args))
  z_bc_flat, unravel = ravel_pytree(z_bc)

  def dynamics_one(t, aug, args):
    y, adj_y, _, _ = aug
    ydot, vjpfun = vjp(f, y, t, args)
    return (ydot, *tree_map(jnp.negative, vjpfun(adj_y)))

  def dynamics_one_flat(t, aug, args):
    flat, _ = ravel_pytree(dynamics_one(t, unravel(aug), args))
    return flat

  @jit
  def dynamics_many_flat(ts, augs, args):
    return vmap(dynamics_one_flat, in_axes=(0, 1, None))(ts, augs, args).T

  def bc(aug_t0, aug_t1):
    y_t0_, _, _, _ = aug_t0
    _, adj_y_t1_, adj_t_t1_, adj_args_t1_ = aug_t1
    return tree_multimap(jnp.subtract, (y_t0_, adj_y_t1_, adj_t_t1_, adj_args_t1_), z_bc)

  @jit
  def bc_flat(aug_t0, aug_t1):
    error_flat, _ = ravel_pytree(bc(unravel(aug_t0), unravel(aug_t1)))
    return error_flat

  dynamics_one_jac = jacrev(dynamics_one_flat, argnums=1)

  @jit
  def dynamics_jac(ts, augs, args):
    return jnp.transpose(vmap(dynamics_one_jac, in_axes=(0, 1, None))(ts, augs, args),
                         axes=(1, 2, 0))

  # If fun_jac isn't provided then the number of nodes blows up, and we reach
  # memory errors, even on a machine with 90G. See the full error for more info:
  # https://gist.github.com/samuela/8c5f6463e08d15c9ffad1f352d1a5513.

  # Adding the bc_jac is super important for numerical stability.
  bvp_soln = solve_bvp(lambda ts, augs: dynamics_many_flat(ts, augs, f_args),
                       bc_flat,
                       jnp.linspace(t0, t1, num=init_num_nodes),
                       jnp.array([z_bc_flat] * init_num_nodes).T,
                       fun_jac=lambda ts, augs: dynamics_jac(ts, augs, f_args),
                       bc_jac=jit(jacrev(bc_flat, argnums=(0, 1))))

  z_t1, _, _, _ = unravel(bvp_soln.y[:, -1])
  _, adj_y_t0, adj_t_t0, adj_args_t0 = unravel(bvp_soln.y[:, 0])
  return z_t1, adj_y_t0, adj_t_t0, adj_args_t0, bvp_soln

def policy_loss_and_grad(dynamics_fn, cost_fn, T, policy):
  def f(y, _t, args):
    _, x = y
    u = policy(args, x)
    xdot = dynamics_fn(x, u)
    return (cost_fn(x, u), xdot)

  def run(x0, args):
    (loss, _), _, _, adj_args_t0, bvp_soln = bvp_fwd_bwd(f, (0.0, x0), 0, T, args,
                                                         (1.0, zeros_like_tree(x0)))
    assert bvp_soln.success
    return loss, adj_args_t0

  return run

def main():
  rng = random.PRNGKey(0)
  x_dim = 2
  T = 20.0

  policy_init, policy = stax.serial(
      Dense(64),
      Tanh,
      Dense(x_dim),
  )

  x0 = jnp.ones(x_dim)

  A, B, Q, R, N = fixed_env(x_dim)
  print("System dynamics:")
  print(f"  A = {A}")
  print(f"  B = {B}")
  print(f"  Q = {Q}")
  print(f"  R = {R}")
  print(f"  N = {N}")
  print()

  dynamics_fn = lambda x, u: A @ x + B @ u
  cost_fn = lambda x, u: x.T @ Q @ x + u.T @ R @ u + 2 * x.T @ N @ u

  ### Evaluate LQR solution to get a sense of optimal cost.
  K, _, _ = control.lqr(A, B, Q, R, N)
  K = jnp.array(K)
  opt_loss, _opt_K_grad = policy_loss_and_grad(dynamics_fn, cost_fn, T, lambda KK, x: -KK @ x)(x0,
                                                                                               K)
  # This is true for longer time horizons, but not true for shorter time
  # horizons due to the LQR solution being an infinite-time solution.
  # assert jnp.allclose(opt_K_grad, 0)

  ### Training loop.
  rng_init_params, rng = random.split(rng)
  _, init_policy_params = policy_init(rng_init_params, (x_dim, ))
  opt = make_optimizer(optimizers.adam(1e-3))(init_policy_params)
  loss_and_grad = policy_loss_and_grad(dynamics_fn, cost_fn, T, policy)

  loss_per_iter = []
  elapsed_per_iter = []
  for iteration in range(10000):
    t0 = time.time()
    loss, g = loss_and_grad(x0, opt.value)
    opt = opt.update(g)
    elapsed = time.time() - t0

    loss_per_iter.append(loss)
    elapsed_per_iter.append(elapsed)

    print(f"Iteration {iteration}")
    print(f"    excess loss = {loss - opt_loss}")
    print(f"    elapsed = {elapsed}")

  blt.remember({
      "loss_per_iter": loss_per_iter,
      "elapsed_per_iter": elapsed_per_iter,
      "opt_loss": opt_loss
  })

  _, ax1 = plt.subplots()
  ax1.set_xlabel("Iteration")
  ax1.set_ylabel("Cost", color="tab:blue")
  ax1.set_yscale("log")
  ax1.tick_params(axis="y", labelcolor="tab:blue")
  ax1.plot(loss_per_iter, color="tab:blue", label="Total rollout cost")
  plt.axhline(opt_loss, linestyle="--", color="gray")
  ax1.legend(loc="upper left")
  plt.title("Combined fwd-bwd BVP problem")
  blt.show()

if __name__ == "__main__":
  main()
