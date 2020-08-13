import PyCall: @pyimport, @py_str, pyimport

# See https://github.com/JuliaPy/PyCall.jl/issues/48#issuecomment-515787405.
py"""
import sys
sys.path.insert(0, "./difftaichi/python")
"""

# @pyimport taichi as ti
@pyimport importlib
# For some reason, @pyimport doesn't work with module reloading.
mass_spring = pyimport("mass_spring")
# include-ing in the REPL should re-import. See https://github.com/JuliaPy/PyCall.jl/issues/611#issuecomment-437625297.
importlib.reload(mass_spring)

mass_spring.main(1; visualize = false)
