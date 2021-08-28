let
  # Last updated: 8/22/21. Check for new commits at status.nixos.org.
  # pkgs = import (fetchTarball("https://github.com/NixOS/nixpkgs/archive/14b0f20fa1f56438b74100513c9b1f7c072cf789.tar.gz")) {};

  # Personal scratch repo
  pkgs = import (fetchTarball("https://github.com/samuela/nixpkgs/archive/769ca7e87aebb24ed34cd7a91854ef2ef186f28f.tar.gz")) {};

  # Rolling updates, not deterministic.
  # pkgs = import (fetchTarball("channel:nixpkgs-unstable")) {};
in pkgs.mkShell {
  buildInputs = with pkgs; [
    ffmpeg
    python3
    python3Packages.botorch
    python3Packages.ipython
    python3Packages.jax
    python3Packages.jaxlib
    python3Packages.matplotlib
    yapf
  ];
}
