include("ppg_toi.jl")

# Training imports
import DiffEqFlux: FastChain, FastDense, initial_params
import DifferentialEquations: solve, Euler, Tsit5, VCABM
import DiffEqSensitivity: ReverseDiffVJP, ZygoteVJP
import Random: seed!
import Flux
import Flux: ADAM, Momentum, Optimiser
import ProgressMeter
import ReverseDiff

# module MassSpringEnv

# MassSpringEnv module imports
import PyCall: @pyimport, @py_str, pyimport
import Statistics: mean
import Zygote

# See https://github.com/JuliaPy/PyCall.jl/issues/48#issuecomment-515787405.
py"""
import sys
sys.path.insert(0, "./difftaichi/python")
"""

np = pyimport("numpy")
importlib = pyimport("importlib")
# For some reason, @pyimport doesn't work with module reloading.
mass_spring = pyimport("mass_spring")
mass_spring_robot_config = pyimport("mass_spring_robot_config")
# include-ing in the REPL should re-import. See https://github.com/JuliaPy/PyCall.jl/issues/611#issuecomment-437625297.
importlib.reload(mass_spring)
importlib.reload(mass_spring_robot_config)

# mass_spring.main(1; visualize = false)
objects, springs = mass_spring_robot_config.robots[4]()
mass_spring.setup_robot(objects, springs)

# The y position of the ground. As defined in the Python version.
ground_height = 0.1

# This was originally in Python (zero-indexed), now in Julia (1-indexed).
head_id = 1

# This apparently is something to do with the sine wave inputs to the policy.
spring_omega = 10

# The number of input sine waves to the policy.
n_sin_waves = 10

# The exact equivalent from the difftaichi code is that our dampening = (1 - exp(-dt * d)) / d where d is their damping.
# In difftaichi d = 15 and dt = 0.004 which comes out to damping = 14.56...
damping = 14.56
# damping = 1 / 0.004
@assert damping >= 0

n_objects = size(objects, 1)
n_springs = size(springs, 1)

# We go back and forth between flat and non-flat representations for x and v. These flattened representations are
# column-major since that's how Julia does things. Note that Numpy is row-major by default however!

# Specifying Array types here is important for ReverseDiff.jl to be able to hit the tracked version of this function
# below. Why is this the case? Who knows. Because x and v are actually ReshapedArrays, it ends up being easier to
# dispatch on u. See https://github.com/JuliaDiff/ReverseDiff.jl/issues/150#issuecomment-692649803.
function forces_fn(x, v, u::Array)
    mass_spring.forces_fn(np.array(x, dtype=np.float32), np.array(v, dtype=np.float32), np.array(u, dtype=np.float32))
end
function forces_fn(x, v, u::ReverseDiff.TrackedArray)
    ReverseDiff.track(forces_fn, x, v, u)
end
ReverseDiff.@grad function forces_fn(x, v, u)
    # We reuse these for both the forward and backward.
    x_np = np.array(x, dtype=np.float32)
    v_np = np.array(v, dtype=np.float32)
    u_np = np.array(u, dtype=np.float32)
    function pullback(ΔΩ)
        mass_spring.forces_fn_vjp(x_np, v_np, u_np, np.array(ΔΩ, dtype=np.float32))
    end
    mass_spring.forces_fn(x_np, v_np, u_np), pullback
end
Zygote.@adjoint function forces_fn(x, v, u)
    # We reuse these for both the forward and backward.
    x_np = np.array(x, dtype=np.float32)
    v_np = np.array(v, dtype=np.float32)
    u_np = np.array(u, dtype=np.float32)
    function pullback(ΔΩ)
        mass_spring.forces_fn_vjp(x_np, v_np, u_np, np.array(ΔΩ, dtype=np.float32))
    end
    mass_spring.forces_fn(x_np, v_np, u_np), pullback
end

import Test: @test
import FiniteDifferences

# begin
#     @info "Testing DiffTaichi gradients"
#     seed!(123)
#     for i in 1:10
#         local x = randn(Float32, (n_objects, 2))
#         local v = randn(Float32, (n_objects, 2))
#         local u = randn(Float32, (n_springs, ))
#         local g = randn(Float32, (n_objects, 2))
#         f(x, v, u) = sum(forces_fn(x, v, u) .* g)
#         local x̄1, v̄1, ū1 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3, 1), f, x, v, u)
#         # local x̄2, v̄2, ū2 = ReverseDiff.gradient(f, (x, v, u))
#         local x̄2, v̄2, ū2 = Zygote.gradient(f, x, v, u)
#         @test isapprox(x̄1, x̄2; rtol=1e-3)
#         @test maximum(abs.(v̄1 - v̄2)) <= 1e-2
#         @test isapprox(ū1, ū2, rtol = 1e-2)
#     end
# end

v_dynamics(v_flat, x_flat, u) = begin
    x = reshape(x_flat, (n_objects, 2))
    v = reshape(v_flat, (n_objects, 2))
    v_acc = forces_fn(x, v, u)
    v_acc -= damping * v

    # Positions are [x, y]. The ground has infinite friction in the difftaichi model.
    # We need to do this in order to work with ReverseDiff.jl since a for-loop requires setindex! which isn't allowed in
    # AD code.
    v_acc_x = [if x[i, 2] > ground_height v_acc[i, 1] else 0.0 end for i in 1:n_objects]
    v_acc_y = [if x[i, 2] > ground_height v_acc[i, 2] else max(0.0, v_acc[i, 2]) end for i in 1:n_objects]

    # We need to flatten everything back down to a vector. Reuse v_flat == v[:].
    [v_acc_x; v_acc_y]
end
x_dynamics(v_flat, x_flat, u) = begin
    v = reshape(v_flat, (n_objects, 2))
    x = reshape(x_flat, (n_objects, 2))
    # This is important so that we don't end up penetrating the floor.
    (v .* (x[:, 2] .> ground_height))[:]
end
# begin
#     @info "Testing dynamics gradients"
#     seed!(123)
#     for i in 1:10
#         local state = 0.01 * randn(Float32, (4 * n_objects, ))
#         local u = 0.01 * randn(Float32, (n_springs, ))
#         local g = 0.01 * randn(Float32, (4 * n_objects, ))
#         f(state, u) = sum(dynamics(state, u) .* g)
#         local g_state_fd, g_u_fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), f, state, u)
#         # local g_state_rd, g_u_rd = ReverseDiff.gradient(f, (state, u))
#         local g_state_rd, g_u_rd = Zygote.gradient(f, state, u)
#         @test isapprox(g_state_fd, g_state_rd; rtol=1e-1)
#         @test maximum(abs.(g_u_fd - g_u_rd)) <= 1e-1
#     end
# end

function cost(v_flat, x_flat, u)
    x = reshape(x_flat, (n_objects, 2))
    -x[head_id, 1]
end

# The name sample_x0 is a hold-over from the other stuff, but really we're getting (v0, x0).
function sample_x0()
    # objects doubles as the initial condition, and we start with zero velocity.
    x0 = np.array(objects)[:]
    (zero(x0), x0)
end

function observation(v_flat, x_flat, t)
    x = reshape(x_flat, (n_objects, 2))

    # Note there is a subtle difference between this and the difftaichi code in
    # that we are doing 1..10, but in Python they do 0..9. It's all just
    # arbitrary inputs to the policy network though, shouldn't make any
    # difference.
    # See https://github.com/JuliaDiff/ReverseDiff.jl/issues/151 for why the
    # `collect` is necessary.
    periodic_signal = sin.(spring_omega * t .+ 2 * π / n_sin_waves .* collect(1:n_sin_waves))

    center = mean(x, dims = 1)
    offsets = x .- center
    [periodic_signal; center[:]; offsets[:]]
end

# begin
#     @info "Testing observation gradients"
#     seed!(123)
#     for i in 1:10
#         local state = 0.1 * randn(Float32, (4 * n_objects, ))
#         local t = randn(Float32)
#         local g = randn(Float32, (n_sin_waves + 2 + 2 * n_objects, ))
#         f(state, t) = sum(observation(state, t) .* g)
#         local g_state_fd, _ = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), f, state, t)
#         local g_state_rd, _ = Zygote.gradient(f, state, t)
#         @test maximum(abs.(g_state_fd - g_state_rd)) <= 1e-3
#     end
# end

################

# Difftaichi trains for 2048 // 3 steps, each being 0.004s long. That all works out to 2.728s.
T = 2.728
num_iters = 10
batch_size = 1
num_hidden = 64
policy = FastChain(
    # We have n_sin_waves scalars, n_objects 2-vectors for each offset, and 1 2-vector for the center.
    FastDense(n_sin_waves + 2 * n_objects + 2, num_hidden, tanh),
    FastDense(num_hidden, num_hidden, tanh),
    FastDense(num_hidden, n_springs),
)
init_policy_params = initial_params(policy)

toi_affect(v_flat, x_flat, dt) = begin
    x = reshape(x_flat, (n_objects, 2))
    v = reshape(v_flat, (n_objects, 2))

    # We want to make sure that we only register for negative velocities.
    tois = [-x[i, 2] / min(v[i, 2], -eps()) for i in 1:n_objects]
    impact_velocities = zero(v)

    # Can't use zip. See https://github.com/FluxML/Zygote.jl/issues/221.
    new_v1 = [if tois[i] < dt impact_velocities[i, 1] else v[i, 1] end for i in 1:n_objects]
    new_v2 = [if tois[i] < dt impact_velocities[i, 2] else v[i, 2] end for i in 1:n_objects]
    new_x1 = x[:, 1] + min.(tois, dt) .* v[:, 1] + max.(dt .- tois, 0) .* new_v1
    new_x2 = x[:, 2] + min.(tois, dt) .* v[:, 2] + max.(dt .- tois, 0) .* new_v2
    ([new_v1; new_v2], [new_x1; new_x2])
end

ppg_loss_pullback = ppg_toi_goodies(
    v_dynamics,
    x_dynamics,
    cost,
    (v_flat, x_flat, params, t) -> policy(observation(v_flat, x_flat, t), params),
    TOIStuff([(v_flat, x_flat) -> reshape(x_flat, (n_objects, 2))[i, 2] - ground_height - 1e-4 for i in 1:n_objects], toi_affect, 1e-6),
    T
)
v0, x0 = sample_x0()
@time fwd_sol, pullback = ppg_loss_pullback(
    v0,
    x0,
    init_policy_params,
    Tsit5(),
    Dict(:rtol => 1e-3, :atol => 1e-3),
    # Dict(:rtol => 1e-3, :atol => 1e-3, :dt => 0.004),
)

# Visualize a rollout:
ts = 0:0.004:T
zs = fwd_sol.(ts)
vs_flat = [z.x[1][2:end] for z in zs]
xs_flat = [z.x[2][2:end] for z in zs]
vs = [reshape(z, (n_objects, 2)) for z in vs_flat]
xs = [reshape(z, (n_objects, 2)) for z in xs_flat]
acts = [policy(observation(v, x, t), init_policy_params) for (v, x, t) in zip(vs_flat, xs_flat, ts)]

# This doesn't work on headless machines.
mass_spring.animate(xs, acts, ground_height, output = "poopypoops")

# @time stuff = pullback(ones(1 + size(sample_x0(), 1)), InterpolatingAdjoint(autojacvec=ZygoteVJP()))

# Train
function run(goodies)
    # Seed here so that both interp and euler get the same batches.
    seed!(123)

    loss_per_iter = fill(NaN, num_iters)
    policy_params_per_iter = fill(NaN, num_iters, length(init_policy_params))
    g_per_iter = fill(NaN, num_iters, length(init_policy_params))
    nf_per_iter = fill(NaN, num_iters)
    n∇ₓf_per_iter = fill(NaN, num_iters)
    n∇ᵤf_per_iter = fill(NaN, num_iters)

    policy_params = deepcopy(init_policy_params)
    # opt = ADAM()
    opt = Momentum(0.001)
    # opt = Optimiser(ExpDecay(0.001, 0.5, 1000, 1e-5), Momentum(0.001))
    # opt =LBFGS(
    #     alphaguess = LineSearches.InitialStatic(alpha = 0.001),
    #     linesearch = LineSearches.Static(),
    # )
    progress = ProgressMeter.Progress(num_iters)
    for iter = 1:num_iters
        x0 = sample_x0()
        sol, pullback = goodies.loss_pullback(
            x0,
            policy_params,
            Euler(),
            # Dict(:rtol => 1e-3, :atol => 1e-3)
            Dict(:dt => 0.004)
        )
        loss = sol.solutions[end].u[end][1]
        @error "yay"
        pb_stuff = pullback([1.0; zero(x0)], InterpolatingAdjoint(autojacvec = ZygoteVJP()))
        g = pb_stuff.g_p

        loss_per_iter[iter] = loss
        policy_params_per_iter[iter, :] = policy_params
        g_per_iter[iter, :] = g
        nf_per_iter[iter] = pb_stuff.nf + sum(s.destats.nf for s in sol.solutions)
        n∇ₓf_per_iter[iter] = pb_stuff.n∇ₓf
        n∇ᵤf_per_iter[iter] = pb_stuff.n∇ᵤf

        # clamp!(g, -10, 10)
        Flux.Optimise.update!(opt, policy_params, g)
        ProgressMeter.next!(
            progress;
            showvalues = [
                (:iter, iter),
                (:loss, loss),
                (:nf, nf_per_iter[iter]),
                (:n∇ₓf, n∇ₓf_per_iter[iter]),
                (:n∇ᵤf, n∇ᵤf_per_iter[iter]),
            ],
        )
    end

    (
        loss_per_iter = loss_per_iter,
        policy_params_per_iter = policy_params_per_iter,
        g_per_iter = g_per_iter,
        nf_per_iter = nf_per_iter,
        n∇ₓf_per_iter = n∇ₓf_per_iter,
        n∇ᵤf_per_iter = n∇ᵤf_per_iter,
    )
end

# @info "InterpolatingAdjoint"
# interp_results = run(learned_policy_goodies)

# import JLSO
# JLSO.save("mass_spring_results.jlso", :results => interp_results)

# mass_spring2 = pyimport("mass_spring2")
# mass_spring2.main(1)

# end

# TODO:
#  * try training with Euler
#  * try training with ppg
