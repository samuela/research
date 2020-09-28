
include("common.jl")
include("../ppg.jl")

using DifferentialEquations #: Tsit5
using Flux
using Flux.Data: DataLoader
using Flux.Optimise: ExpDecay
using DiffEqFlux: FastChain, FastDense, initial_params
using Random: seed!
#using Plots
using Statistics: mean
using DiffEqSensitivity #: InterpolatingAdjoint, BacksolveAdjoint, QuadratureAdjoint
using JLSO
using Optim #: LBFGS
using LineSearches
using ProgressMeter
using Statistics

using LinearAlgebra
BLAS.set_num_threads(1)

# This seeds the `init_policy_params` below, and then gets overridden later.
seed!(12345)

floatT = Float32
T = 5.0
num_iters = 200
batch_size = 16 #32
#dstate = 12
#dact = 4

dynamics, cost, sample_x0, obs = QuadrotorEnv.normalenv(floatT, 9.8f0, 3.0f0,
                                                        1.0f0, 1.0f0, 1.0f0)
dobs = length(obs(sample_x0()))
dstate = length(sample_x0())

function newpolicy()
    num_hidden = 16 #64
    act = tanh
    FastChain(
              (x, _) -> obs(x),
              FastDense(dobs, num_hidden, act),
              FastDense(num_hidden, num_hidden, act),
              FastDense(num_hidden, 4,
                        initW=(x...)->Flux.glorot_uniform(x...)*1e-1
                       ),
             )
end

policy = newpolicy()


rtol = 1e-3
atol = 1e-3

lpg = ppg_goodies(dynamics, cost, policy, T; reltol=rtol, abstol=atol)

function run(loss_and_grad, p0, sample_env, rseed)
    # Seed here so that both interp and euler get the same batches.
    seed!(rseed)

    loss_per_iter = fill(NaN, num_iters)
    policy_params_per_iter = fill(NaN, num_iters, length(p0))
    g_per_iter = fill(NaN, num_iters, length(p0))
    nf_per_iter = fill(NaN, num_iters)
    n∇f_per_iter = fill(NaN, num_iters)

    policy_params = deepcopy(p0)
    batches = [[sample_env() for _ = 1:batch_size] for _ = 1:num_iters]
    opt = ADAM(0.01)
    #opt = Momentum(0.0001)
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
        n∇f_per_iter[iter] = info.n∇ₓf + info.n∇ᵤf

        clamp!(g, -10, 10)
        Flux.Optimise.update!(opt, policy_params, g)
        ProgressMeter.next!(
                            progress;
                            showvalues = [
                                          (:iter, iter),
                                          (:loss, loss),
                                          (:nf, info.nf / batch_size),
                                          (:n∇f, (info.n∇ₓf + info.n∇ᵤf) / batch_size),
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

########### normal model
N = 5
interp_times = zeros(N)
euler_times = zeros(N)
for i=1:N
    init_params = initial_params(newpolicy()) # get new policy parameters, randomly
    rseed = 12345 + 1000*i

    solver = VCABM
    @info "Interp"
    interp_times[i] = @elapsed interp_results = run(
                         (x0_batch, θ) -> lpg.ez_loss_and_grad_many(
                                                                    x0_batch,
                                                                    θ,
                                                                    solver(),
                                                                    InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
                                                                    #InterpolatingAdjoint(),
                                                                    #QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))
                                                                   ),
                         init_params, sample_x0, rseed
                        )

    # Divide by two for the forward and backward passes.
    mean_euler_steps =
    mean((interp_results.nf_per_iter + interp_results.n∇f_per_iter) / batch_size / 2)
    euler_dt = T / mean_euler_steps
    #euler_dt = 0.05

    @info "Euler"
    @info euler_dt
    euler_times[i] = @elapsed euler_results = run(
                                           (x0_batch, θ) ->
                                           lpg.ez_euler_loss_and_grad_many(x0_batch, θ, euler_dt),
                                           init_params, sample_x0, rseed
                                          )

    @info "Dumping results"
    JLSO.save(
              "quadrotor_train_results_$i.jlso",
              :interp_results => interp_results,
              :euler_results  => euler_results,
             )
    @info "Interp time avg: $(mean(interp_times)), std: $(std(interp_times))"
    @info "Euler time avg: $(mean(euler_times)), std: $(std(euler_times))"

    using UnicodePlots
    xmax = maximum(vcat(sum(euler_results.nf_per_iter) +sum(euler_results.n∇f_per_iter),
                        sum(interp_results.nf_per_iter)+sum(interp_results.n∇f_per_iter)))

    ymin, ymax = round.(extrema([euler_results.loss_per_iter; interp_results.loss_per_iter]))

    plt = lineplot(cumsum(euler_results.nf_per_iter + euler_results.n∇f_per_iter),
                   euler_results.loss_per_iter,
                   name = "Euler BPTT, normal",
                   xlabel = "Number of function evaluations", ylabel = "Loss",
                   color = :blue,
                   ylim=(ymin, ymax),
                   xlim=(0,xmax),
                   width=80, height=14
                  )
    lineplot!(plt,
              cumsum(
                     interp_results.nf_per_iter + interp_results.n∇f_per_iter,
                    ),
              interp_results.loss_per_iter,
              name = "PPG (ours), normal",
              color = :red
             )
    display(plt)


    plt = lineplot(1:num_iters,
                   euler_results.loss_per_iter,
                   name = "Euler BPTT, normal",
                   xlabel = "Number of iterations", ylabel = "Loss",
                   color = :blue,
                   ylim=(ymin, ymax),
                   xlim=(0,num_iters),
                   width=80, height=14
                  )
    lineplot!(plt,
              1:num_iters,
              interp_results.loss_per_iter,
              name = "PPG (ours), normal",
              color = :red
             )
    display(plt)
end

#=
plt = lineplot(1:num_iters,
cumsum(euler_results.nf_per_iter + euler_results.n∇f_per_iter),
name = "Euler BPTT, normal",
xlabel = "Iterations", ylabel = "Function Evals",
color = :blue,
ylim=(0, 500000),
xlim=(0, 20),
width=80, height=14
)
lineplot!(plt,
1:num_iters,
cumsum(
interp_results.nf_per_iter + interp_results.n∇f_per_iter,
),
name = "PPG (ours), normall",
color = :red
)
display(plt)
=#

