import functools
from multiprocessing import get_context
import os
from pathlib import Path
import pickle

import tqdm
import numpy as np
from jax import random

from research.estop.half_cheetah import config, run_ddpg

# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

num_episodes = 1000

def job(random_seed: int, base_dir: Path):
  rng = random.PRNGKey(random_seed)
  np.random.seed(random_seed)

  callback_rng, train_rng = random.split(rng)
  callback_rngs = random.split(callback_rng, num_episodes)

  params = [None]
  tracking_params = [None]
  train_reward_per_episode = []
  policy_value_per_episode = []
  elapsed_per_episode = []

  def callback(info):
    episode = info['episode']
    params[0] = info["optimizer"].value
    tracking_params[0] = info["tracking_params"]

    train_reward_per_episode.append(info["reward"])
    elapsed_per_episode.append(info["elapsed"])

    if episode % run_ddpg.policy_evaluation_frequency == 0:
      policy_value = run_ddpg.eval_policy(callback_rngs[episode],
                                          info["optimizer"].value[0])
      policy_value_per_episode.append(policy_value)

  run_ddpg.train(
      train_rng,
      num_episodes,
      lambda t, _: t >= config.episode_length,
      callback,
  )
  with (base_dir / f"seed={random_seed}.pkl").open(mode="wb") as f:
    pickle.dump(
        {
            "final_params": params[0],
            "final_tracking_params": tracking_params[0],
            "train_reward_per_episode": train_reward_per_episode,
            "policy_value_per_episode": policy_value_per_episode,
            "elapsed_per_episode": elapsed_per_episode,
        }, f)

def main():
  num_random_seeds = 72

  # Create necessary directory structure.
  results_dir = Path("results/ddpg_half_cheetah")
  full_results_dir = results_dir / "full"
  results_dir.mkdir()
  full_results_dir.mkdir()

  pickle.dump(
      {
          "gamma": config.gamma,
          "episode_length": config.episode_length,
          "num_random_seeds": num_random_seeds,
          "num_episodes": num_episodes,
          "tau": run_ddpg.tau,
          "buffer_size": run_ddpg.buffer_size,
          "batch_size": run_ddpg.batch_size,
      }, (results_dir / "metadata.pkl").open(mode="wb"))

  # See https://codewithoutrules.com/2018/09/04/python-multiprocessing/.
  with get_context("spawn").Pool() as pool:
    for _ in tqdm.tqdm(pool.imap_unordered(
        functools.partial(job, base_dir=full_results_dir),
        range(num_random_seeds)),
                       desc="full",
                       total=num_random_seeds):
      pass

if __name__ == "__main__":
  main()
