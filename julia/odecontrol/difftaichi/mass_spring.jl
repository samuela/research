import DifferentialEquations: solve, Euler

# module MassSpringEnv

import PyCall: @pyimport, @py_str, pyimport

# See https://github.com/JuliaPy/PyCall.jl/issues/48#issuecomment-515787405.
py"""
import sys
sys.path.insert(0, "./difftaichi/python")
"""

# @pyimport taichi as ti
@pyimport numpy as np
@pyimport importlib
# For some reason, @pyimport doesn't work with module reloading.
mass_spring = pyimport("mass_spring")
mass_spring_robot_config = pyimport("mass_spring_robot_config")
# include-ing in the REPL should re-import. See https://github.com/JuliaPy/PyCall.jl/issues/611#issuecomment-437625297.
importlib.reload(mass_spring)
importlib.reload(mass_spring_robot_config)

# mass_spring.main(1; visualize = false)
objects, springs = mass_spring_robot_config.robots[2]()
mass_spring.setup_robot(objects, springs)

# This was originally in Python (zero-indexed), now in Julia (1-indexed).
head_id = 1

# This apparently is something to do with the sine wave inputs to the policy.
spring_omega = 10

# The number of input sine waves to the policy.
n_sin_waves = 10

n_objects = length(objects)
function dynamics(state, u)
    x_flat = @view state[1:2*n_objects]
    v_flat = @view state[2*n_objects+1:end]
    x = reshape(x_flat, (n_objects, 2))
    v = reshape(v_flat, (n_objects, 2))
    v_acc = mass_spring.forces_fn(x, v, u)
    # We need to flatten everything back down to a vector. Reuse v_flat == v[:].
    [v_flat; v_acc[:]]
end

function cost(state, u)
    x_flat = @view state[1:2*n_objects]
    x = reshape(x_flat, (n_objects, 2))
    -x[head_id, 1]
end

function sample_x0()
    # objects doubles as the initial condition, and we start with zero velocity.
    x0 = np.array(objects)[:]
    [x0; zero(x0)]
end

function observation(state, t)
    x_flat = @view state[1:2*n_objects]
    v_flat = @view state[2*n_objects+1:end]
    x = reshape(x_flat, (n_objects, 2))
    v = reshape(v_flat, (n_objects, 2))

    # Note there is a subtle difference between this and the difftaichi code in
    # that we are doing 1..10, but in Python they do 0..9. It's all just
    # arbitrary inputs to the policy network though, shouldn't make any
    # difference.
    periodic_signal = sin.(spring_omega * t + 2 * π / n_sin_waves .* (1:n_sin_waves))

    center = mean(x, dims=1)
    offsets = x .- center
    [periodic_signal; center; offsets[:]]
end

mass_spring.animate(100)

# mass_spring2 = pyimport("mass_spring2")
# mass_spring2.main(1)

# end
