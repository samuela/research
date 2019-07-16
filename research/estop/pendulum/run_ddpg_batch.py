import os

import functools
from multiprocessing import get_context
from pathlib import Path
import pickle

import tqdm
from jax import random

from research.estop.pendulum import config, run_ddpg

# Limit ourselves to single-threaded numpy operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

def job(random_seed: int, base_dir: Path):
  seed_dir = base_dir / f"seed={random_seed}"
  seed_dir.mkdir()

  def callback(info):
    episode = info["episode"]
    if episode % 100 == 0:
      with (seed_dir / f"episode={episode}.pkl").open(mode="wb") as f:
        pickle.dump(info["optimizer"].value, f)

  res = run_ddpg.train(random.PRNGKey(random_seed), callback)
  with (seed_dir / "reward_per_episode.pkl").open(mode="wb") as f:
    pickle.dump(res["reward_per_episode"], f)
  with (seed_dir / "final_params.pkl").open(mode="wb") as f:
    pickle.dump(res["optimizer"].value, f)

def main():
  num_random_seeds = 100

  # Create necessary directory structure.
  results_dir = Path("results/ddpg_pendulum")
  full_results_dir = results_dir / "full"
  results_dir.mkdir()
  full_results_dir.mkdir()

  pickle.dump(
      {
          "gamma": config.gamma,
          "episode_length": config.episode_length,
          "max_torque": config.max_torque,
          "num_random_seeds": num_random_seeds,
          "num_episodes": run_ddpg.num_episodes,
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
