import pickle

from jax import random
import jax.numpy as jp

from research.estop import ddpg
from research.estop.pendulum import config
from research.estop.pendulum.run_ddpg import actor
from research.statistax import Deterministic

experiment_folder = "7_8ade325_ddpg_pendulum"
seed = 4
episode = -1
num_rollouts = 1000

print("Loading pkl...")
data = pickle.load(
    open(f"results/{experiment_folder}/full/seed={seed}.pkl", "rb"))
actor_params, _ = data["params_per_episode"][episode]

print("Rolling out trajectories...")
rng = random.PRNGKey(0)
rollout_rngs = random.split(rng, num_rollouts)
states_list = [
    ddpg.rollout(
        rollout_rngs[i],
        config.env,
        lambda s: Deterministic(actor(actor_params, s)),
        num_timesteps=1000,
    )[0] for i in range(num_rollouts)
]
states = jp.concatenate(states_list, axis=0)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(states[:, 0], states[:, 1], alpha=0.1)
# plt.title(f"Seed {seed}")
# plt.show()

pickle.dump(
    {
        "experiment_folder": experiment_folder,
        "seed": seed,
        "episode": episode,
        "num_rollouts": num_rollouts,
        "states": states,
    }, open("results/support_set.pkl", "wb"))
