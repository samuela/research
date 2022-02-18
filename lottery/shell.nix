let
  # Last updated: 1/27/2022. Check for new commits at status.nixos.org.
  # pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/83ab260bbe7e4c27bb1f467ada0265ba13bbeeb0.tar.gz") { };
  # pkgs = import (/home/skainswo/dev/nixpkgs) { };

  # Differences on top of nixpkgs mainline:
  # - https://github.com/NixOS/nixpkgs/pull/158218 merged in
  # - https://github.com/samuela/nixpkgs/commit/cedb9abbb1969073f3e6d76a68da8835ec70ddb0 updates jaxlib-bin to use the cuDNN 8.3 instead of 8.1 to get around https://github.com/google/jax/discussions/9455
  pkgs = import (fetchTarball "https://github.com/samuela/nixpkgs/archive/cedb9abbb1969073f3e6d76a68da8835ec70ddb0.tar.gz") { };
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    ffmpeg
    python3
    python3Packages.flax
    python3Packages.ipython
    python3Packages.jax
    # See https://discourse.nixos.org/t/petition-to-build-and-cache-unfree-packages-on-cache-nixos-org/17440/14
    # as to why we don't use the source builds of jaxlib/tensorflow.
    (python3Packages.jaxlib-bin.override {
      cudaSupport = true;
    })
    python3Packages.matplotlib
    python3Packages.plotly
    (python3Packages.tensorflow-bin.override {
      cudaSupport = true;
    })
    python3Packages.tensorflow-datasets
    python3Packages.tqdm
    python3Packages.wandb
    yapf
  ];
}
