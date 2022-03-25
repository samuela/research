# Run with nixGL, eg `nixGLNvidia-510.47.03 python cifar10_convnet_run.py --test`

let
  # pkgs = import (/home/skainswo/dev/nixpkgs) { };

  # Last updated: 2022-03-07. Check for new commits at status.nixos.org.
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/4d60081494259c0785f7e228518fee74e0792c1b.tar.gz") {
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
}
