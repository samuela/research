import functools
from multiprocessing import cpu_count, get_context
import os
from pathlib import Path
import pickle

import tqdm
import numpy as np
from jax import jit, random

from research.estop.gym.half_cheetah import spec
from research.estop.gym.half_cheetah.ddpg import run as run_ddpg

# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

def job(
    random_seed: int,
    num_episodes: int,
    state_min,
    state_max,
    out_dir: Path,
):
  job_dir = out_dir / f"seed={random_seed}"
  job_dir.mkdir()

  rng = random.PRNGKey(random_seed)
  np.random.seed(random_seed)

  callback_rng, train_rng = random.split(rng)
  callback_rngs = random.split(callback_rng, num_episodes)

  params = [None]
  tracking_params = [None]
  discounted_cumulative_reward_per_episode = []
  undiscounted_cumulative_reward_per_episode = []
  policy_evaluations = []
  episode_lengths = []
  elapsed_per_episode = []

  def callback(info):
    episode = info['episode']
    params[0] = info["optimizer"].value
    tracking_params[0] = info["tracking_params"]

    current_actor_params, _ = info["optimizer"].value

    discounted_cumulative_reward_per_episode.append(
        info["discounted_cumulative_reward"])
    undiscounted_cumulative_reward_per_episode.append(
        info["undiscounted_cumulative_reward"])
    episode_lengths.append(info["episode_length"])
    elapsed_per_episode.append(info["elapsed"])

    if episode % run_ddpg.policy_evaluation_frequency == 0:
      curr_policy = jit(run_ddpg.deterministic_policy(current_actor_params))
      policy_value = run_ddpg.eval_policy(callback_rngs[episode], curr_policy)
      policy_evaluations.append(policy_value)

    if (episode + 1) % run_ddpg.policy_video_frequency == 0:
      curr_policy = jit(run_ddpg.deterministic_policy(current_actor_params))
      run_ddpg.film_policy(callback_rngs[episode],
                           curr_policy,
                           filepath=job_dir / f"episode_{episode}.mp4")

  run_ddpg.train(
      train_rng,
      num_episodes,
      lambda t, s: ((t >= spec.max_episode_steps) or np.any(s < state_min) or np
                    .any(s > state_max)),
      callback,
  )
  with (job_dir / f"data.pkl").open(mode="wb") as f:
    pickle.dump(
        {
            "final_params": params[0],
            "final_tracking_params": tracking_params[0],
            "discounted_cumulative_reward_per_episode":
            discounted_cumulative_reward_per_episode,
            "undiscounted_cumulative_reward_per_episode":
            undiscounted_cumulative_reward_per_episode,
            "policy_evaluations": policy_evaluations,
            "episode_lengths": episode_lengths,
            "elapsed_per_episode": elapsed_per_episode,
            "state_min": state_min,
            "state_max": state_max,
        }, f)

def main():
  num_random_seeds = 48
  num_episodes = 10000

  # Create necessary directory structure.
  results_dir = Path("results/ddpg_half_cheetah")
  results_dir.mkdir()

  pickle.dump(
      {
          "type": "vanilla",
          "gamma": run_ddpg.gamma,
          "episode_length": spec.max_episode_steps,
          "num_random_seeds": num_random_seeds,
          "num_episodes": num_episodes,
          "tau": run_ddpg.tau,
          "buffer_size": run_ddpg.buffer_size,
          "batch_size": run_ddpg.batch_size,
          "num_eval_rollouts": run_ddpg.num_eval_rollouts,
          "policy_evaluation_frequency": run_ddpg.policy_evaluation_frequency,
      }, (results_dir / "metadata.pkl").open(mode="wb"))

  # See https://codewithoutrules.com/2018/09/04/python-multiprocessing/.
  # Running a single job usually takes up about 1.5-2 cores since mujoco runs
  # separately and we can't really control its parallelism.
  with get_context("spawn").Pool(processes=cpu_count() // 2) as pool:
    for _ in tqdm.tqdm(pool.imap_unordered(
        functools.partial(job,
                          out_dir=results_dir,
                          num_episodes=num_episodes,
                          state_min=-np.inf * np.ones(spec.state_shape),
                          state_max=np.inf * np.ones(spec.state_shape)),
        range(num_random_seeds)),
                       total=num_random_seeds):
      pass

if __name__ == "__main__":
  main()
