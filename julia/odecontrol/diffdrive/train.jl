"""Train a differential drive policy and create an animation of the training
process displaying its adaptation on a set of paths over time. Note that when
running on a headless machine, the environment variable `GKS_WSTYPE=140`
generally needs to be set. See https://discourse.julialang.org/t/unable-to-display-plot-using-the-repl-gks-errors/12826/16.
"""

include("common.jl")
include("../ppg.jl")

import DifferentialEquations: Tsit5
import Flux
import Flux: ADAM
import Flux.Data: DataLoader
import DiffEqFlux: FastChain, FastDense, initial_params
import Random: seed!
import Plots
import Statistics: mean
import DiffEqSensitivity: InterpolatingAdjoint, BacksolveAdjoint, QuadratureAdjoint
import JLSO

# This seeds the `init_policy_params` below, and then gets overridden later.
seed!(123)

floatT = Float32
T = 5.0
num_iters = 10000
batch_size = 32

dynamics, cost, sample_x0, obs = DiffDriveEnv.diffdrive_env(floatT, 1.0f0, 0.5f0)

num_hidden = 32
policy = FastChain(
    (x, _) -> obs(x),
    FastDense(7, num_hidden, tanh),
    FastDense(num_hidden, num_hidden, tanh),
    FastDense(num_hidden, 2),
)
# policy = FastDense(x_dim, x_dim) # linear policy

init_policy_params = initial_params(policy) * 0.1
learned_policy_goodies = ppg_goodies(dynamics, cost, policy, T)

function interp()
    # Seed here so that both interp and euler get the same batches.
    seed!(123)

    loss_per_iter = fill(NaN, num_iters)
    policy_params_per_iter = fill(NaN, num_iters, length(init_policy_params))
    nf_per_iter = fill(NaN, num_iters)
    n∇f_per_iter = fill(NaN, num_iters)

    policy_params = deepcopy(init_policy_params)
    opt = ADAM()
    for iter = 1:num_iters
        @time begin
            x0_batch = [sample_x0() for _ = 1:batch_size]
            loss, g, info = learned_policy_goodies.ez_loss_and_grad_many(
                x0_batch,
                policy_params,
                InterpolatingAdjoint(),
            )
            loss_per_iter[iter] = loss
            policy_params_per_iter[iter, :] = policy_params
            nf_per_iter[iter] = info.nf
            n∇f_per_iter[iter] = info.n∇f

            Flux.Optimise.update!(opt, policy_params, g)
            println("Episode $iter, loss = $loss")
        end
    end

    (
        loss_per_iter = loss_per_iter,
        policy_params_per_iter = policy_params_per_iter,
        nf_per_iter = nf_per_iter,
        n∇f_per_iter = n∇f_per_iter,
    )
end

@info "Interp"
interp_results = interp()

# Divide by two for the forward and backward passes.
mean_euler_steps =
    mean((interp_results.nf_per_iter + interp_results.n∇f_per_iter) / batch_size / 2)
euler_dt = T / mean_euler_steps

function euler(dt)
    # Seed here so that both interp and euler get the same batches.
    seed!(123)

    loss_per_iter = fill(NaN, num_iters)
    policy_params_per_iter = fill(NaN, num_iters, length(init_policy_params))
    nf_per_iter = fill(NaN, num_iters)
    n∇f_per_iter = fill(NaN, num_iters)

    policy_params = deepcopy(init_policy_params)
    opt = ADAM()
    for iter = 1:num_iters
        @time begin
            x0_batch = [sample_x0() for _ = 1:batch_size]
            loss, g, info = learned_policy_goodies.ez_euler_loss_and_grad_many(
                x0_batch,
                policy_params,
                dt,
            )
            loss_per_iter[iter] = loss
            policy_params_per_iter[iter, :] = policy_params
            nf_per_iter[iter] = info.nf
            n∇f_per_iter[iter] = info.n∇f

            Flux.Optimise.update!(opt, policy_params, g)
            println("Episode $iter, loss = $loss")
        end
    end

    (
        loss_per_iter = loss_per_iter,
        policy_params_per_iter = policy_params_per_iter,
        nf_per_iter = nf_per_iter,
        n∇f_per_iter = n∇f_per_iter,
    )
end

@info "Euler"
euler_results = euler(euler_dt)

@info "Dumping results"
JLSO.save(
    "diffdrive_train_results.jlso",
    :interp_results => interp_results,
    :euler_results => euler_results,
)
