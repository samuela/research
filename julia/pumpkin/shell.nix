# nix-shell --run "julia --eval 'import Pluto; Pluto.run(; host=\"0.0.0.0\", require_secret_for_access=false, launch_browser=false, auto_reload_from_file=true)'"

let
  # Last updated: 1/11/22. From status.nixos.org.
  pkgs = import (fetchTarball ("https://github.com/NixOS/nixpkgs/archive/81f05d871faf75d1456df6adec1d2118d787f65c.tar.gz")) { };

  # Rolling updates, not deterministic.
  # pkgs = import (fetchTarball("channel:nixpkgs-unstable")) {};
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    # See https://github.com/NixOS/nixpkgs/issues/66716. Necessary for julia to
    # be able to download packages.
    cacert

    julia_17-bin

    python3
    python3Packages.flax
    python3Packages.jax
    python3Packages.jaxlib
    python3Packages.optax
    python3Packages.tqdm
    python3Packages.wandb

    yapf
  ];

  # See https://github.com/JuliaPy/PyCall.jl/issues/952#issuecomment-1005694327
  PYTHON = "${pkgs.python3}/bin/python";
}
