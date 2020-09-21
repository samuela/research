include("../ppg.jl")

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
import DifferentialEquations: VectorContinuousCallback

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

# The exact equivalent from the difftaichi code is that our dampening = (1 - exp(-dt * d)) / d where d is their damping.
# In difftaichi d = 15 and dt = 0.004 which comes out to damping = 14.56...
damping = 14.56

n_objects = length(objects)
n_springs = size(springs, 1)

# We go back and forth between flat and non-flat representations for x and v. These flattened representations are
# column-major since that's how Julia does things. Note that Numpy is row-major by default however!

import Zygote

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

begin
    @info "Testing DiffTaichi gradients"
    seed!(123)
    for i in 1:10
        local x = randn(Float32, (n_objects, 2))
        local v = randn(Float32, (n_objects, 2))
        local u = randn(Float32, (n_springs, ))
        local g = randn(Float32, (n_objects, 2))
        f(x, v, u) = sum(forces_fn(x, v, u) .* g)
        local x̄1, v̄1, ū1 = FiniteDifferences.grad(FiniteDifferences.central_fdm(3, 1), f, x, v, u)
        # local x̄2, v̄2, ū2 = ReverseDiff.gradient(f, (x, v, u))
        local x̄2, v̄2, ū2 = Zygote.gradient(f, x, v, u)
        @test isapprox(x̄1, x̄2; rtol=1e-3)
        @test maximum(abs.(v̄1 - v̄2)) <= 1e-2
        @test isapprox(ū1, ū2, rtol = 1e-2)
    end
end

function dynamics(state, u)
    x_flat = @view state[1:2*n_objects]
    v_flat = @view state[2*n_objects+1:end]
    x = reshape(x_flat, (n_objects, 2))
    v = reshape(v_flat, (n_objects, 2))
    # TODO: did the original difftaichi example damp gravity as well? We are.
    v_acc = forces_fn(x, v, u)
    v_acc -= damping * v

    # Positions are [x, y]. The ground has infinite friction in the difftaichi model.
    # We need to do this in order to work with ReverseDiff.jl since a for-loop requires setindex! which isn't allowed in
    # AD code.
    v_acc_x = [if x[i, 2] > ground_height v_acc[i, 1] else 0.0 end for i in 1:n_objects]
    v_acc_y = [if x[i, 2] > ground_height v_acc[i, 2] else max(0.0, v_acc[i, 2]) end for i in 1:n_objects]

    # We need to flatten everything back down to a vector. Reuse v_flat == v[:].
    [v_flat; v_acc_x; v_acc_y]
end

begin
    @info "Testing dynamics gradients"
    seed!(123)
    for i in 1:10
        local state = 0.01 * randn(Float32, (4 * n_objects, ))
        local u = 0.01 * randn(Float32, (n_springs, ))
        local g = 0.01 * randn(Float32, (4 * n_objects, ))
        f(state, u) = sum(dynamics(state, u) .* g)
        local g_state_fd, g_u_fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), f, state, u)
        # local g_state_rd, g_u_rd = ReverseDiff.gradient(f, (state, u))
        local g_state_rd, g_u_rd = Zygote.gradient(f, state, u)
        @test isapprox(g_state_fd, g_state_rd; rtol=1e-1)
        @test maximum(abs.(g_u_fd - g_u_rd)) <= 1e-1
    end
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
    # See https://github.com/JuliaDiff/ReverseDiff.jl/issues/151 for why the
    # `collect` is necessary.
    periodic_signal = sin.(spring_omega * t .+ 2 * π / n_sin_waves .* collect(1:n_sin_waves))

    center = mean(x, dims = 1)
    offsets = x .- center
    [periodic_signal; center[:]; offsets[:]]
end

begin
    @info "Testing observation gradients"
    seed!(123)
    for i in 1:10
        local state = 0.1 * randn(Float32, (4 * n_objects, ))
        local t = randn(Float32)
        local g = randn(Float32, (n_sin_waves + 2 + 2 * n_objects, ))
        f(state, t) = sum(observation(state, t) .* g)
        local g_state_fd, _ = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), f, state, t)
        local g_state_rd, _ = Zygote.gradient(f, state, t)
        @test maximum(abs.(g_state_fd - g_state_rd)) <= 1e-3
    end
end

################

callback = VectorContinuousCallback(
    (out, u, _, _) -> begin
        state = @view u[2:end]
        x_flat = @view state[1:2*n_objects]
        x = reshape(x_flat, (n_objects, 2))
        out[:] .= x[:, 2] .- ground_height
    end,
    (integrator, idx) -> begin
        # TODO: this arithmetic is disgusting. Use RecursiveArrayTools or something else.
        # Don't forget the 1 to account for the cost.
        # Set the y to ground_height.
        integrator.u[1+n_objects+idx] = ground_height
        # Set the x velocity to zero.
        integrator.u[1+2*n_objects+idx] = 0
        # Set the y velocity to zero.
        integrator.u[1+3*n_objects+idx] = 0
    end,
    n_objects,
    abstol=1e-2
)

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
init_policy_params = 0.1 * initial_params(policy)

learned_policy_goodies =
    ppg_goodies(dynamics, cost, (x, t, params) -> policy(observation(x, t), params), T)
@time fwd_sol, pullback = learned_policy_goodies.loss_pullback(
    sample_x0(),
    init_policy_params,
    Tsit5(),
    Dict(:callback => callback, :rtol => 1e-3, :atol => 1e-3),
    # Dict(:callback => callback, :rtol => 1e-3, :atol => 1e-3, :dt => 0.004),
)
@time stuff = pullback(ones(1 + size(sample_x0(), 1)), InterpolatingAdjoint(autojacvec=ZygoteVJP()))

batch_size = 1

# Train
function run(loss_and_grad)
    # Seed here so that both interp and euler get the same batches.
    seed!(123)

    loss_per_iter = fill(NaN, num_iters)
    policy_params_per_iter = fill(NaN, num_iters, length(init_policy_params))
    g_per_iter = fill(NaN, num_iters, length(init_policy_params))
    nf_per_iter = fill(NaN, num_iters)
    n∇ₓf_per_iter = fill(NaN, num_iters)
    n∇ᵤf_per_iter = fill(NaN, num_iters)

    policy_params = deepcopy(init_policy_params)
    batches = [[sample_x0() for _ = 1:batch_size] for _ = 1:num_iters]
    # opt = ADAM()
    opt = Momentum(0.001)
    # opt = Optimiser(ExpDecay(0.001, 0.5, 1000, 1e-5), Momentum(0.001))
    # opt =LBFGS(
    #     alphaguess = LineSearches.InitialStatic(alpha = 0.001),
    #     linesearch = LineSearches.Static(),
    # )
    progress = ProgressMeter.Progress(num_iters)
    for iter = 1:num_iters
        loss, g, info = loss_and_grad(batches[iter], policy_params)
        loss_per_iter[iter] = loss
        policy_params_per_iter[iter, :] = policy_params
        g_per_iter[iter, :] = g
        nf_per_iter[iter] = info.nf
        n∇ₓf_per_iter[iter] = info.n∇ₓf
        n∇ᵤf_per_iter[iter] = info.n∇ᵤf

        # clamp!(g, -10, 10)
        Flux.Optimise.update!(opt, policy_params, g)
        ProgressMeter.next!(
            progress;
            showvalues = [
                (:iter, iter),
                (:loss, loss),
                (:nf, info.nf / batch_size),
                (:n∇ₓf, info.n∇ₓf / batch_size),
                (:n∇ᵤf, info.n∇ᵤf / batch_size),
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

@info "InterpolatingAdjoint"
interp_results = run(
    (x0_batch, θ) -> learned_policy_goodies.ez_loss_and_grad_many(
        x0_batch,
        θ,
        VCABM(),
        InterpolatingAdjoint(autojacvec = ZygoteVJP()),
        fwd_solve_kwargs = Dict(:callback => callback, :rtol => 1e-3, :atol => 1e-3),
    ),
)

import JLSO
JLSO.save("mass_spring_results.jlso", :results => interp_results)

# Visualize a rollout:
ts = 0:0.01:T
zs = fwd_sol.(ts)
xs_flat = [z[2:end] for z in zs]
xs = [reshape(z[1:2*n_objects], (n_objects,2)) for z in xs_flat]
acts = [policy(observation(x, t), interp_results.policy_params_per_iter[end, :]) for (x, t) in zip(xs_flat, ts)]

# This doesn't work on headless machines.
mass_spring.animate(xs, acts, ground_height, output = "poopypoops")

# mass_spring2 = pyimport("mass_spring2")
# mass_spring2.main(1)

# end

# TODO:
#  * try training with Euler
#  * try training with ppg
