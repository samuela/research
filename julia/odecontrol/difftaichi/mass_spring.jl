include("../ppg.jl")

import DiffEqFlux: FastChain, FastDense, initial_params

import DifferentialEquations: solve, Euler, Tsit5

# module MassSpringEnv

import PyCall: @pyimport, @py_str, pyimport
import Statistics: mean

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

# The y position of the ground. As defined in the Python version.
ground_height = 0.1

# This was originally in Python (zero-indexed), now in Julia (1-indexed).
head_id = 1

# This apparently is something to do with the sine wave inputs to the policy.
spring_omega = 10

# The number of input sine waves to the policy.
n_sin_waves = 10

n_objects = length(objects)
n_springs = size(springs, 1)

# We go back and forth between flat and non-flat representations for x and v. These flattened representations are
# column-major since that's how Julia does things. Note that Numpy is row-major by default however!

function dynamics(state, u)
    x_flat = @view state[1:2*n_objects]
    v_flat = @view state[2*n_objects+1:end]
    x = reshape(x_flat, (n_objects, 2))
    v = reshape(v_flat, (n_objects, 2))
    v_acc = mass_spring.forces_fn(np.array(x), np.array(v), u)

    for i in 1:n_objects
        # positions are [x, y]. The ground has infinite friction in the difftaichi model.
        if x[i, 2] <= ground_height
            v_acc[i, 1] = 0
            v_acc[i, 2] = max(0, v_acc[i, 2])
        end
    end

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
    periodic_signal = sin.(spring_omega * t .+ 2 * π / n_sin_waves .* (1:n_sin_waves))

    center = mean(x, dims=1)
    offsets = x .- center
    [periodic_signal; center[:]; offsets[:]]
end

###

T = 10.0
num_hidden = 64
policy = FastChain(
    # We have n_sin_waves scalars, n_objects 2-vectors for each offset, and 1 2-vector for the center.
    FastDense(n_sin_waves + 2 * n_objects + 2, num_hidden, tanh),
    FastDense(num_hidden, num_hidden, tanh),
    FastDense(num_hidden, n_springs),
)

init_policy_params = initial_params(policy)
goodies = ppg_goodies(dynamics, cost, (x, t, params) -> policy(observation(x, t), params), T)
fwd_sol, _ = goodies.loss_pullback(sample_x0(), init_policy_params, Tsit5())

ts = 0:0.01:T
zs = fwd_sol.(ts)
xs_flat = [z[2:end] for z in zs]
xs = [reshape(z[1:2*n_objects], (n_objects,2)) for z in xs_flat]
acts = [policy(observation(x, t), init_policy_params) for (x, t) in zip(xs_flat, ts)]

mass_spring.animate(xs, acts, ground_height, output = "poopypoops")


# mass_spring2 = pyimport("mass_spring2")
# mass_spring2.main(1)

# end
