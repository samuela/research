let
  # Last updated: 8/22/21. Check for new commits at status.nixos.org.
  # pkgs = import (fetchTarball("https://github.com/NixOS/nixpkgs/archive/14b0f20fa1f56438b74100513c9b1f7c072cf789.tar.gz")) {};

  # Personal scratch repo. Needed to get jax, jaxlib. See https://github.com/NixOS/nixpkgs/pull/134894.
  pkgs = import (fetchTarball("https://github.com/samuela/nixpkgs/archive/6dd6095f47930e1db75f7b61ec917c5aebc83446.tar.gz")) {};

  # Rolling updates, not deterministic.
  # pkgs = import (fetchTarball("channel:nixpkgs-unstable")) {};
in pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.ipython
    python3Packages.jax
    (python3Packages.jaxlib.override { cudaSupport = true; })
    python3Packages.matplotlib
    python3Packages.wandb
    yapf
  ];

  # See https://github.com/google/jax/issues/5723#issuecomment-913038780
  XLA_FLAGS = "--xla_gpu_force_compilation_parallelism=1";
}
