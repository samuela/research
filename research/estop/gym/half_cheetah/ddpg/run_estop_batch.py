import functools
from multiprocessing import cpu_count, get_context
import os
from pathlib import Path
import pickle

import tqdm
import numpy as np
from jax import jit, random

from research.estop.half_cheetah import config, run_ddpg, run_ddpg_batch

# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

input_results_dir = Path("results/12_4de1834_ddpg_half_cheetah")
output_results_dir = Path("results/estop_ddpg_half_cheetah")
num_support_set_rollouts = 500
num_random_seeds = 48
num_episodes = 10000
tolerance = 0

if __name__ == "__main__":
  rng = random.PRNGKey(0)

  print("Loading vanilla DDPG results...")
  experiment_metadata = pickle.load(
      (input_results_dir / "metadata.pkl").open("rb"))

  data = [
      pickle.load((input_results_dir / f"seed={seed}" / "data.pkl").open("rb"))
      for seed in range(experiment_metadata["num_random_seeds"])
  ]
  final_policy_values = np.array([x["policy_evaluations"][-1] for x in data])
  best_seed = int(np.argmax(final_policy_values))
  print(
      f"... best seed is {best_seed} with cumulative reward: {data[best_seed]['policy_evaluations'][-1]}"
  )

  print("Rolling out trajectories from best policy...")
  actor_params, _ = data[best_seed]["final_params"]
  expert_policy = jit(run_ddpg.deterministic_policy(actor_params))
  support_set_rollouts = np.array([
      run_ddpg.rollout(r, expert_policy)[0]
      for r in tqdm.tqdm(random.split(rng, num_support_set_rollouts))
  ])

  state_min = np.min(support_set_rollouts, axis=(0, 1))
  state_max = np.max(support_set_rollouts, axis=(0, 1))

  ###

  # Create necessary directory structure.
  output_results_dir.mkdir()

  pickle.dump(
      {
          "type": "estop",
          "gamma": config.gamma,
          "episode_length": config.episode_length,
          "num_random_seeds": num_random_seeds,
          "num_episodes": num_episodes,
          "tau": run_ddpg.tau,
          "buffer_size": run_ddpg.buffer_size,
          "batch_size": run_ddpg.batch_size,
          "num_eval_rollouts": run_ddpg.num_eval_rollouts,
          "policy_evaluation_frequency": run_ddpg.policy_evaluation_frequency,

          # E-stop specific
          "tolerance": tolerance,
          "num_support_set_rollouts": num_support_set_rollouts,
          "state_min": state_min,
          "state_max": state_max,
      },
      (output_results_dir / "metadata.pkl").open(mode="wb"))

  # See https://codewithoutrules.com/2018/09/04/python-multiprocessing/.
  # Running a single job usually takes up about 1.5-2 cores since mujoco runs
  # separately and we can't really control its parallelism.
  with get_context("spawn").Pool(processes=cpu_count() // 2) as pool:
    for _ in tqdm.tqdm(pool.imap_unordered(
        functools.partial(run_ddpg_batch.job,
                          num_episodes=num_episodes,
                          base_dir=output_results_dir,
                          state_min=state_min - tolerance,
                          state_max=state_max + tolerance),
        range(num_random_seeds)),
                       total=num_random_seeds):
      pass
