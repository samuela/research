let
  # Last updated: 8/28/21. From status.nixos.org.
  pkgs = import (fetchTarball("https://github.com/NixOS/nixpkgs/archive/503209808cd613daed238e21e7a18ffcbeacebe3.tar.gz")) {};

  # Rolling updates, not deterministic.
  # pkgs = import (fetchTarball("channel:nixpkgs-unstable")) {};
in pkgs.mkShell {
  buildInputs = with pkgs; [
    # See https://github.com/NixOS/nixpkgs/issues/66716. Necessary for julia to
    # be able to download packages.
    cacert

    ffmpeg
    julia_16-bin
    python3
    python3Packages.pip
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
