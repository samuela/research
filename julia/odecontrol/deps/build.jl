"""PyCall has its own little Python install, and we need to make sure that we
install all the right dependencies in that environment. Note that you need to
restart Julia after running this in order to be able to @pyimport the installed
packages. Run this with `] build`.

See https://gist.github.com/Luthaf/368a23981c8ec095c3eb."""

import PyCall: @pyimport

@pyimport pip
const PIP_PACKAGES = ["taichi", "scipy", "matplotlib"]
pip.main(["install", "--user", PIP_PACKAGES...])
