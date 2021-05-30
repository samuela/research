let
  # Last updated: 4/26/21. From status.nixos.org.
  pkgs = import (fetchTarball("https://github.com/NixOS/nixpkgs/archive/32f7980afb5e33f1e078a51e715b9f102f396a69.tar.gz")) {};

  # We can get rid of this once https://github.com/NixOS/nixpkgs/pull/117881 merges.
  pkgs_with_julia = import (fetchTarball("https://github.com/NixOS/nixpkgs/archive/b82a21edaedb1b3687d5dce63a433f56456c80e8.tar.gz")) {};

  # Rolling updates, not deterministic.
  # pkgs = import (fetchTarball("channel:nixpkgs-unstable")) {};
in pkgs.mkShell {
  buildInputs = [
    pkgs.python3
    pkgs.python3.pkgs.pip
    pkgs_with_julia.julia_16
    pkgs.ffmpeg
  ];
  shellHook = ''
    # Hacks to make taichi work:
    # See https://nixos.wiki/wiki/Packaging/Quirks_and_Caveats#ImportError:_libstdc.2B.2B.so.6:_cannot_open_shared_object_file:_No_such_file.
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib/:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${pkgs.xorg.libX11}/lib/:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${pkgs.ncurses5.out}/lib/:$LD_LIBRARY_PATH"

    # Tells pip to put packages into $PIP_PREFIX instead of the usual locations.
    # See https://pip.pypa.io/en/stable/user_guide/#environment-variables.
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH
  '';
}
