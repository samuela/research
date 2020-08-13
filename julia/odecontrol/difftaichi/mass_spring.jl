import PyCall: @pyimport, @py_str

# See https://github.com/JuliaPy/PyCall.jl/issues/48#issuecomment-515787405.
py"""
import sys
sys.path.insert(0, "./difftaichi/python")
"""

# @pyimport taichi as ti
@pyimport mass_spring
