import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from pathlib import Path
import pickle

import numpy as np

def plot_errorfill(x, ys, color):
  mean = np.mean(ys, axis=0)
  std = np.std(ys, axis=0)
  #   plt.plot(x, mean, color=color, alpha=1.0)
  plt.plot(x, np.median(ys, axis=0), color=color, alpha=1.0)
  plt.fill_between(x,
                   np.maximum(mean - std, ys.min(axis=0)),
                   np.minimum(mean + std, ys.max(axis=0)),
                   color=color,
                   alpha=0.25)

if __name__ == "__main__":
  results_dir = Path("results/qlearning_pkls_444a731")

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
  # This is not actually the number of episodes since we only record once every
  # few episodes.
  num_episodes_to_plot = 2500
  num_points_to_plot = int(num_episodes_to_plot / policy_evaluation_frequency)

  plt.figure()
  x = policy_evaluation_frequency * np.arange(num_points_to_plot)
  plot_errorfill(x, full_policy_rewards[:, :num_points_to_plot], "slategrey")
  plot_errorfill(x, estop_policy_rewards[:, :num_points_to_plot], "crimson")
  plt.axhline(optimal_policy_reward, color="grey", linestyle="--")
  plt.legend(["Full env. Q-learning", "E-stop Q-learning", "Optimal policy"])
  plt.xlabel("Episode")
  plt.ylabel("Policy reward")
  plt.tight_layout()
  plt.savefig("figs/q_learning_per_episode.pdf")

  ### Plot per states seen.
  # Plotting every single state is really overkill on this scale and makes
  # vector graphics huge and sluggish.
  num_states_seen_to_plot = 550000
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
  plot_errorfill(x, full_policy_rewards_interp, "slategrey")
  plot_errorfill(x, estop_policy_rewards_interp, "crimson")
  plt.axhline(optimal_policy_reward, color="grey", linestyle="--")
  plt.legend(["Full env. Q-learning", "E-stop Q-learning", "Optimal policy"])
  plt.xlabel("Number of states seen")
  plt.ylabel("Policy reward")
  plt.tight_layout()
  plt.savefig("figs/q_learning_per_states_seen.pdf")

  plt.show()
