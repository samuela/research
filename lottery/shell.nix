let
  # Last updated: 1/6/2022. Check for new commits at status.nixos.org.
  # Can't use mainline until https://github.com/NixOS/nixpkgs/pull/153761 lands.
  # pkgs = import (fetchTarball ("https://github.com/NixOS/nixpkgs/archive/77fda7f672726e1a95c8cd200f27bccfc86c870b.tar.gz")) { };

  # Personal scratch repo. Needed to get jax, jaxlib. See https://github.com/NixOS/nixpkgs/pull/134894.
  pkgs = import (fetchTarball ("https://github.com/samuela/nixpkgs/archive/72a96723ae1f7e3c7a7fa486e5c9160b4145d58a.tar.gz")) { };

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
