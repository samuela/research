import pickle

from jax import random

from research.estop import ddpg
from research.estop.pendulum import config
from research.estop.pendulum.env import viz_pendulum_rollout
from research.estop.pendulum.run_ddpg import actor
from research.statistax import Deterministic

experiment_folder = "5_15ba7ed_ddpg_pendulum"
seed = 0
episode = -1

data = pickle.load(
    open(f"results/{experiment_folder}/full/seed={seed}.pkl", "rb"))
actor_params, _ = data["params_per_episode"][episode]

rng = random.PRNGKey(0)
while True:
  rollout_rng, rng = random.split(rng)
  states, actions = ddpg.rollout(
      rollout_rng,
      config.env,
      lambda s: Deterministic(actor(actor_params, s)),
      num_timesteps=250,
  )
  viz_pendulum_rollout(states, 2 * actions / config.max_torque)
