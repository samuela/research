let
  # Last updated: 1/27/2022. Check for new commits at status.nixos.org.
  # pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/83ab260bbe7e4c27bb1f467ada0265ba13bbeeb0.tar.gz") { };
  # pkgs = import (/home/skainswo/dev/nixpkgs) { };

  # See https://nixos.wiki/wiki/FAQ/Pinning_Nixpkgs.
  pkgs =
    let
      unstablePkgsSrc = fetchTarball "https://github.com/NixOS/nixpkgs/archive/f97413a248ef6a6ac03ac789d0234d67f0aba0f0.tar.gz";
      unstablePkgs = import unstablePkgsSrc { };
      patches = [
        # cudnn_cudatoolkit_11: 8.1.1 -> 8.3.0
        (unstablePkgs.fetchpatch {
          url = "https://github.com/NixOS/nixpkgs/pull/158218.patch";
          sha256 = "0fcd92ya5yahakmvkv17rfcfbz79p16arpyn5w9r988yvkh2n5xx";
        })
        # python3Packages.jaxlib-bin: 0.1.71 -> 0.1.75
        (unstablePkgs.fetchpatch {
          url = "https://github.com/NixOS/nixpkgs/pull/158186.patch";
          sha256 = "191xh2z89r28a3mx0m5fk4jfhhnzi53vi7b44gknbkhip673xrbg";
        })
        # TODO: create a PR for this one, once #158186 is merged.
        # python3Packages.jaxlib-bin: cudnn805 -> cudnn82
        (unstablePkgs.fetchpatch {
          url = "https://github.com/samuela/nixpkgs/commit/214f1a49dd533b37d2a94400dd20844156353245.patch";
          sha256 = "07zp03l6v9hkw2zncalm88hkp0kxl05xbyrdf29mfvchma2ncd8f";
        })
        # TODO: PR is merged, so this one should no longer be necessary once we
        # bump the version of nixpkgs.
        # python3Packages.wandb: 0.12.9 -> 0.12.10
        (unstablePkgs.fetchpatch {
          url = "https://github.com/NixOS/nixpkgs/pull/157788.patch";
          sha256 = "0h1qsv47ywg3j0nhdhcrr1f70hi3fsdf51z5c7z48124k9463wvy";
        })
      ];
      patchedPkgsSrc =
        if (builtins.length patches == 0) then
          unstablePkgsSrc else
          unstablePkgs.runCommand "patched-nixpkgs"
            { inherit unstablePkgsSrc; inherit patches; }
            ''
              cp -r $unstablePkgsSrc $out
              chmod -R +w $out
              for p in $patches; do
                echo "Applying patch $p";
                patch -d $out -p1 < "$p";
              done
            '';
    in
    import patchedPkgsSrc { };
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
