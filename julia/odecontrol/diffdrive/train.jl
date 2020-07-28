"""Train a differential drive policy and create an animation of the training
process displaying its adaptation on a set of paths over time. Note that when
running on a headless machine, the environment variable `GKS_WSTYPE=140`
generally needs to be set. See https://discourse.julialang.org/t/unable-to-display-plot-using-the-repl-gks-errors/12826/16.
"""

include("common.jl")
include("../ppg.jl")

using DiffEqFlux, DiffEqSensitivity, DifferentialEquations, Flux
using Flux.Data, Flux.Optimise
using JLSO
using Random
using Random: seed!
#using Plots
using Statistics
using Optim, LineSearches
using ProgressMeter
using UnicodePlots

using LinearAlgebra
BLAS.set_num_threads(1)

# This seeds the `init_policy_params` below, and then gets overridden later.
seed!(123)

floatT = Float32
T = 5.0f0 #10.0
num_iters = 100
batch_size = 32 #8 #128 #32 #8

dynamics, cost, sample_x0, obs = DiffDriveEnv.diffdrive_env(floatT, 1.0f0, 0.5f0)
#const dd = DDrive{floatT}(1.0f0, 0.5f0)

num_hidden = 512 #32
act = tanh #relu
policy = FastChain(
                   (x, _) -> obs(x),
                   FastDense(7, num_hidden, act),
                   FastDense(num_hidden, num_hidden, act),
                   FastDense(num_hidden, 2,
                             initW=(x...)->Flux.glorot_uniform(x...).*1e-2
                            ),
)

# linear policy
#policy = FastChain((x, _) -> obs(x), FastDense(7, 2,
#                                               initW=(x...)->zeros(Float32, x...)))

init_policy_params = initial_params(policy)
lpg = ppg_goodies(dynamics, cost, policy, T)

function run(loss_and_grad, num_iters=num_iters)
    # Seed here so that both interp and euler get the same batches.
    #seed!(123)

    policy_params_per_iter = fill(NaN, num_iters, length(init_policy_params))
    loss_per_iter = fill(NaN, num_iters)
    g_per_iter    = fill(NaN, num_iters, length(init_policy_params))
    nf_per_iter   = fill(NaN, num_iters)
    n∇f_per_iter  = fill(NaN, num_iters)

    policy_params = deepcopy(init_policy_params)
    batches = [[sample_x0() for _ = 1:batch_size] for _ = 1:num_iters]
    #opt = ADAM(0.01)
    opt = Momentum(0.003)
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
        n∇f_per_iter[iter] = info.n∇f

        #clamp!(g, -1, 1)
        Flux.Optimise.update!(opt, policy_params, g)
        ProgressMeter.next!(
                            progress;
                            showvalues = [
                                          (:iter, iter),
                                          (:loss, loss),
                                          (:nf, info.nf / batch_size),
                                          (:n∇f, info.n∇f / batch_size),
                                         ],
                           )
    end

    (
     loss_per_iter = loss_per_iter,
     policy_params_per_iter = policy_params_per_iter,
     g_per_iter = g_per_iter,
     nf_per_iter = nf_per_iter,
     n∇f_per_iter = n∇f_per_iter,
    )
end

nrandom = 2
for i=1:nrandom
    #seed!(i*100 + 12345)
    seed!(i + 12345)
    @info "Interp"
    @time interp_results = run(
                               (x0_batch, θ) ->
                               lpg.ez_loss_and_grad_many(x0_batch, θ,
                                                         InterpolatingAdjoint()),
                              )

    # Divide by two for the forward and backward passes.
    mean_euler_steps =
    mean((interp_results.nf_per_iter + interp_results.n∇f_per_iter) / batch_size / 2)
    euler_dt = T / mean_euler_steps
    #euler_dt = 0.01
    #euler_dt = 0.05
    println("EULER DT: ", euler_dt)

    seed!(i + 12345)
    @info "Euler"
    @time euler_results = run(
                              (x0_batch, θ) ->
                              lpg.ez_euler_loss_and_grad_many(x0_batch, θ,
                                                              euler_dt),
                              num_iters
                             )

    @info "Dumping results"
    JLSO.save(
              "diffdrive_train_results_$i.jlso",
              #:neural_ode_results => neural_ode_results,
              :interp_results => interp_results,
              :euler_results => euler_results,
             )

    xmax = max(sum(euler_results.nf_per_iter)+sum(euler_results.n∇f_per_iter),
               sum(interp_results.nf_per_iter)+sum(interp_results.n∇f_per_iter))
    #xmax = mean([sum(euler_results.nf_per_iter)+sum(euler_results.n∇f_per_iter),
    #             sum(interp_results.nf_per_iter)+sum(interp_results.n∇f_per_iter)])
    plt = lineplot(cumsum(euler_results.nf_per_iter + euler_results.n∇f_per_iter),
                   euler_results.loss_per_iter,
                   name = "Euler BPTT",
                   xlabel = "Number of function evaluations", ylabel = "Loss",
                   color = :blue,
                   #ylim=(20, 200),
                   xlim=(0,xmax),
                   width=80, height=14
                  )
    lineplot!(plt,
              cumsum(
                     interp_results.nf_per_iter + interp_results.n∇f_per_iter,
                    ),
              interp_results.loss_per_iter,
              name = "PPG (ours)",
              color = :red
             )
    display(plt)
end
