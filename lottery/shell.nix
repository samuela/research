# Run with nixGL, eg `nixGLNvidia-510.47.03 python cifar10_convnet_run.py --test`

let
  # Last updated: 1/27/2022. Check for new commits at status.nixos.org.
  # pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/83ab260bbe7e4c27bb1f467ada0265ba13bbeeb0.tar.gz") { };
  # pkgs = import (/home/skainswo/dev/nixpkgs) { };

  # Differences on top of nixpkgs mainline:
  # - https://github.com/samuela/nixpkgs/commit/cedb9abbb1969073f3e6d76a68da8835ec70ddb0 updates jaxlib-bin to use the cuDNN 8.3 instead of 8.1 to get around https://github.com/google/jax/discussions/9455
  # TODO overlay to override cudatoolkit with cudatoolkit 11.5, etc.
  pkgs = import (fetchTarball "https://github.com/samuela/nixpkgs/archive/4ef4292aef7a236fdc84d097ac9086bd45ec8ba3.tar.gz") {
    config.allowUnfree = true;
    # These actually cause problems for some reason. bug report?
    # config.cudaSupport = true;
    # config.cudnnSupport = true;
  };
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    ffmpeg
    python3
    python3Packages.augmax
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
      cudaSupport = false;
    })
    python3Packages.tensorflow-datasets
    python3Packages.tqdm
    python3Packages.wandb
    yapf
  ];

  # See
  #  * https://discourse.nixos.org/t/using-cuda-enabled-packages-on-non-nixos-systems/17788
  #  * https://discourse.nixos.org/t/cuda-from-nixkgs-in-non-nixos-case/7100
  #  * https://github.com/guibou/nixGL/issues/50
  #
  # Note that we just do our best to stay up to date with whatever the latest cudatoolkit version is, and hope that it's
  # compatible with what's used in jaxlib-bin. See https://github.com/samuela/nixpkgs/commit/cedb9abbb1969073f3e6d76a68da8835ec70ddb0#commitcomment-67106407.
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.cudatoolkit_11_5}/lib
  '';
}
