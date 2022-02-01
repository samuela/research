let
  # Last updated: 1/27/2022. Check for new commits at status.nixos.org.
  # pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/83ab260bbe7e4c27bb1f467ada0265ba13bbeeb0.tar.gz") { };
  # pkgs = import (/home/skainswo/dev/nixpkgs) { };

  # See https://nixos.wiki/wiki/FAQ/Pinning_Nixpkgs.
  pkgs =
    let
      unstablePkgsSrc = fetchTarball "https://github.com/NixOS/nixpkgs/archive/376934f4b7ca6910b243be5fabcf3f4228043725.tar.gz";
      unstablePkgs = import unstablePkgsSrc { };
      patches = [
        # For example,
        # (unstablePkgs.fetchpatch {
        #   url = "https://github.com/NixOS/nixpkgs/pull/157055.patch";
        #   sha256 = "05sgxajdka9k1np0clsdl3i9iil4hbkc5zkzp09dv7c2fv545wmi";
        # })
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
    python3
    python3Packages.flax
    python3Packages.ipython
    python3Packages.jax
    python3Packages.jaxlibWithCuda
    python3Packages.matplotlib
    python3Packages.plotly
    python3Packages.tensorflowWithCuda
    python3Packages.tensorflow-datasets
    python3Packages.tqdm
    python3Packages.wandb
    yapf
  ];
}
