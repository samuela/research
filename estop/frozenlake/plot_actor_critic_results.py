import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from pathlib import Path
import pickle

import numpy as np

import viz

if __name__ == "__main__":
  results_dir = Path("results/actor_critic_pkls_e9c2c7c")

  metadata = pickle.load((results_dir / "metadata.pkl").open(mode="rb"))
  num_random_seeds = metadata["num_random_seeds"]
  policy_evaluation_frequency = metadata["policy_evaluation_frequency"]
  optimal_policy_reward = metadata["optimal_policy_reward"]

  estop_results = [
      pickle.load((results_dir / "estop" / f"seed={seed}.pkl").open(mode="rb"))
      for seed in range(num_random_seeds)
  ]
  full_results = [
      pickle.load((results_dir / "full" / f"seed={seed}.pkl").open(mode="rb"))
      for seed in range(num_random_seeds)
  ]

  estop_policy_rewards = np.array(
      [run["policy_rewards"] for run in estop_results])
  full_policy_rewards = np.array(
      [run["policy_rewards"] for run in full_results])

  estop_states_seen = np.array([run["states_seen"] for run in estop_results])
  full_states_seen = np.array([run["states_seen"] for run in full_results])

  ### Plot per episode.
  plt.rcParams.update({"font.size": 16})

  plt.figure()
  x = policy_evaluation_frequency * np.arange(full_policy_rewards.shape[1])
  viz.plot_errorfill(x, full_policy_rewards, "slategrey")
  viz.plot_errorfill(x, estop_policy_rewards, "crimson")
  plt.axhline(optimal_policy_reward, color="grey", linestyle="--")
  plt.legend(
      ["Full env. Actor-Critic", "E-stop Actor-Critic", "Optimal policy"])
  plt.xlabel("Episode")
  plt.ylabel("Cumulative policy reward")
  plt.tight_layout()
  plt.savefig("figs/actor_critic_per_episode.pdf")

  ### Plot per states seen.
  # Plotting every single state is really overkill on this scale and makes
  # vector graphics huge and sluggish.
  num_states_seen_to_plot = 5e6
  freq = 1000
  x = freq * np.arange(int(num_states_seen_to_plot / freq))
  estop_policy_rewards_interp = np.array([
      np.interp(x,
                estop_states_seen[i, :],
                estop_policy_rewards[i, :],
                right=estop_policy_rewards[i, -1])
      for i in range(num_random_seeds)
  ])
  full_policy_rewards_interp = np.array([
      np.interp(x,
                full_states_seen[i, :],
                full_policy_rewards[i, :],
                right=full_policy_rewards[i, -1])
      for i in range(num_random_seeds)
  ])

  plt.figure()
  viz.plot_errorfill(x / 1000, full_policy_rewards_interp, "slategrey")
  viz.plot_errorfill(x / 1000, estop_policy_rewards_interp, "crimson")
  plt.axhline(optimal_policy_reward, color="grey", linestyle="--")
  plt.legend(
      ["Full env. Actor-Critic", "E-stop Actor-Critic", "Optimal policy"],
      loc="lower right")
  plt.xlabel("Number of states seen (thousands)")
  plt.ylabel("Cumulative policy reward")
  plt.tight_layout()
  plt.savefig("figs/actor_critic_per_states_seen.pdf")