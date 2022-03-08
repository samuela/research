# Run with nixGL, eg `nixGLNvidia-510.47.03 python cifar10_convnet_run.py --test`

let
  # pkgs = import (/home/skainswo/dev/nixpkgs) { };

  # Last updated: 2022-03-07. Check for new commits at status.nixos.org.
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/1fc7212a2c3992eedc6eedf498955c321ad81cc2.tar.gz") {
    config.allowUnfree = true;
    # These actually cause problems for some reason. bug report?
    # config.cudaSupport = true;
    # config.cudnnSupport = true;

    # Note that this overlay currently doesn't really accomplish much since we override jaxlib-bin CUDA dependencies.
    overlays = [
      (final: prev: {
        cudatoolkit = prev.cudatoolkit_11_5;
        cudnn = prev.cudnn_8_3_cudatoolkit_11_5;
        # blas = prev.blas.override { blasProvider = final.mkl; };
        # lapack = prev.lapack.override { lapackProvider = final.mkl; };
      })
    ];
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
      cudatoolkit_11 = cudatoolkit_11_5;
      cudnn = cudnn_8_3_cudatoolkit_11_5;
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
