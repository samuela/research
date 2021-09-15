let
  # Last updated: 8/22/21. Check for new commits at status.nixos.org.
  # pkgs = import (fetchTarball("https://github.com/NixOS/nixpkgs/archive/14b0f20fa1f56438b74100513c9b1f7c072cf789.tar.gz")) {};

  # Personal scratch repo. Needed to get jax, jaxlib. See https://github.com/NixOS/nixpkgs/pull/134894.
  pkgs = import (fetchTarball ("https://github.com/samuela/nixpkgs/archive/607934fba6f291bffbfe3ab4d633e7f57e974c5e.tar.gz")) { };

  # Rolling updates, not deterministic.
  # pkgs = import (fetchTarball("channel:nixpkgs-unstable")) {};
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.flax
    python3Packages.ipython
    python3Packages.jax
    (python3Packages.jaxlib.override { cudaSupport = true; })
    python3Packages.matplotlib
    python3Packages.tqdm
    python3Packages.wandb
    yapf
  ];
}
