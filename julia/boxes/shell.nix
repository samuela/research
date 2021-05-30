let
  # Last updated: 5/28/21. From status.nixos.org.
  pkgs = import (fetchTarball("https://github.com/NixOS/nixpkgs/archive/84aa23742f6c72501f9cc209f29c438766f5352d.tar.gz")) {};

  # Rolling updates, not deterministic.
  # pkgs = import (fetchTarball("channel:nixpkgs-unstable")) {};
in pkgs.mkShell {
  buildInputs = with pkgs; [
    julia_16-bin
  ];
}
