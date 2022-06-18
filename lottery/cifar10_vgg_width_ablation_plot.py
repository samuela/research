import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import wandb

if __name__ == "__main__":
  matplotlib.rcParams["font.family"] = "serif"
  matplotlib.rcParams["font.size"] = 12

  api = wandb.Api()
  wm8_run = api.run("skainswo/playing-the-lottery/i85jxkgf")
  wm16_run = api.run("skainswo/playing-the-lottery/gk003bct")
  wm32_run = api.run("skainswo/playing-the-lottery/1np06qag")
  wm64_run = api.run("skainswo/playing-the-lottery/31kskp4e")
  wm128_run = api.run("skainswo/playing-the-lottery/17huxv8g")
  all_runs = [wm8_run, wm16_run, wm32_run, wm64_run, wm128_run]

  fig = plt.figure()
  ax = fig.add_subplot(111)
  lambdas = np.linspace(0, 1, 25)
  wm_glyphs = ["⅛", "¼", "½", "1", "2"]
  cmap = plt.get_cmap("YlOrRd")
  for i, wm_glyph, run in zip(range(len(all_runs)), wm_glyphs, all_runs):
    ys = np.array(run.summary["train_loss_interp_clever"])
    ys = ys - 0.5 * (ys[0] + ys[-1])
    ax.plot(lambdas,
            ys,
            color=cmap(0.25 + 0.75 * i / len(all_runs)),
            linewidth=2,
            label=f"{wm_glyph}× width")
  ax.set_xlabel("$\lambda$")
  ax.set_xticks([0, 1])
  ax.set_xticklabels(["Model $A$", "Model $B$"])
  ax.set_ylabel("Normalized training loss")
  ax.set_title(f"CIFAR-10 VGG-16 Width Ablation")
  ax.legend(loc="upper right", framealpha=0.5)
  fig.tight_layout()

  fig.savefig("cifar10_vgg_width_ablation_plot_train_loss.png", dpi=300)
  fig.savefig("cifar10_vgg_width_ablation_plot_train_loss.eps")

  ############
  fig = plt.figure()
  ax = fig.add_subplot(111)
  lambdas = np.linspace(0, 1, 25)
  wm_glyphs = ["⅛", "¼", "½", "1", "2"]
  cmap = plt.get_cmap("YlOrRd")
  for i, wm_glyph, run in zip(range(len(all_runs)), wm_glyphs, all_runs):
    ys = np.array(run.summary["test_loss_interp_clever"])
    ys = ys - 0.5 * (ys[0] + ys[-1])
    ax.plot(lambdas,
            ys,
            color=cmap(0.25 + 0.75 * i / len(all_runs)),
            linewidth=2,
            label=f"{wm_glyph}× width")
  ax.set_xlabel("$\lambda$")
  ax.set_xticks([0, 1])
  ax.set_xticklabels(["Model $A$", "Model $B$"])
  ax.set_ylabel("Normalized test loss")
  ax.set_title(f"CIFAR-10 VGG-16 Width Ablation")
  ax.legend(loc="upper right", framealpha=0.5)
  fig.tight_layout()

  fig.savefig("cifar10_vgg_width_ablation_plot_test_loss.png", dpi=300)
  fig.savefig("cifar10_vgg_width_ablation_plot_test_loss.eps")
