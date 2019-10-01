from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from research.estop.frozenlake import viz

full_results_dir = Path("results/12_4de1834_ddpg_half_cheetah")
# estop_results_dir = Path("results/10_58d6605_estop_ddpg_pendulum")
num_random_seeds = 48

def load_policy_evaluations(results_dir):
  metadata = pickle.load((full_results_dir / "metadata.pkl").open(mode="rb"))
  assert num_random_seeds == metadata["num_random_seeds"]
  all_seeds = [
      pickle.load((results_dir / f"seed={seed}" / "data.pkl").open(mode="rb"))
      for seed in range(num_random_seeds)
  ]

  policy_evaluation_frequency = metadata["policy_evaluation_frequency"]

  episode_lengths = np.array([x["episode_lengths"] for x in all_seeds])
  steps_seen = np.cumsum(episode_lengths, axis=-1)

  policy_evaluations = np.array([x["policy_evaluations"] for x in all_seeds])

  return steps_seen[:, ::policy_evaluation_frequency], policy_evaluations

if __name__ == "__main__":
  print("Loading full results...")
  full_steps_seen, full_policy_evaluations = load_policy_evaluations(
      full_results_dir)
  print("... done")

  #   print("Loading e-stop results...")
  #   estop_episode_lengths, estop_policy_values = load_policy_evaluations(
  #       estop_results_dir)
  #   estop_steps_seen = np.cumsum(estop_episode_lengths, axis=-1)
  #   print("... done")

  num_steps_seen_to_plot = 1000 * 10000
  freq = 100
  x = freq * np.arange(int(num_steps_seen_to_plot / freq))
  #   estop_policy_values_interp = np.array([
  #       np.interp(x,
  #                 estop_steps_seen[i, :],
  #                 estop_policy_values[i, :],
  #                 right=estop_policy_values[i, -1])
  #       for i in range(num_random_seeds)
  #   ])
  full_policy_values_interp = np.array([
      np.interp(x,
                full_steps_seen[i, :],
                full_policy_evaluations[i, :],
                right=full_policy_evaluations[i, -1])
      for i in range(num_random_seeds)
  ])

  plt.figure()
  viz.plot_errorfill(x / 1000, full_policy_values_interp, "slategrey")
  # viz.plot_errorfill(x / 1000, estop_policy_values_interp, "crimson")
  # plt.legend(["Full env. DDPG", "E-stop DDPG"])
  plt.xlabel("Number of steps seen (thousands)")
  plt.ylabel("Cumulative policy reward")
  plt.tight_layout()
  plt.savefig("figs/deleteme_full_vs_estop_ddpg_half_cheetah.pdf")

  plt.show()
