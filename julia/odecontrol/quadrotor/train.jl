"""Train a differential drive policy and create an animation of the training
process displaying its adaptation on a set of paths over time. Note that when
running on a headless machine, the environment variable `GKS_WSTYPE=140`
generally needs to be set. See https://discourse.julialang.org/t/unable-to-display-plot-using-the-repl-gks-errors/12826/16.
"""

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

num_hidden = 16 #64
act = tanh
policy = FastChain(
    (x, _) -> obs(x),
    FastDense(dobs, num_hidden, act),
    FastDense(num_hidden, num_hidden, act),
    FastDense(num_hidden, 4,
              initW=(x...)->Flux.glorot_uniform(x...)*1e-2
             ),
)
# linear policy
# policy = FastChain((x, _) -> obs(x), FastDense(7, 2))
init_policy_params = initial_params(policy)

############### next model
rdynamics, rcost, rsample_x0, robs = QuadrotorEnv.residualenv(floatT, 9.8f0, 3.0f0,
                                                          1.0f0, 1.0f0, 1.0f0)
dobs = length(robs(rsample_x0()))

rpolicy = FastChain(
    (x, _) -> robs(x),
    FastDense(dobs, num_hidden, act),
    FastDense(num_hidden, num_hidden, act),
    FastDense(num_hidden, 4,
              initW=(x...)->Flux.glorot_normal(x...)*1e-2
             ),
)
res_policy_params = initial_params(rpolicy)

rtol = 1e-3
atol = 1e-3

lpg = ppg_goodies(dynamics, cost, policy, T; reltol=rtol, abstol=atol)
rpg = ppg_goodies(rdynamics, rcost, rpolicy, T; reltol=rtol, abstol=atol)

function run(loss_and_grad, p0, sample_env)
    # Seed here so that both interp and euler get the same batches.
    seed!(123)

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

function rollout(x0, dyn, polparams)
    z0  = [zero(eltype(x0)); x0]
    fwd_sol = solve(
                    ODEProblem(dyn, z0, (0, T), polparams),
                    solver(),
                    u0 = z0,
                    p = polparams,
                    reltol = rtol,
                    abstol = atol,
                   )
    fwd_sol
end

########### normal model
solver = VCABM
@info "Interp"
interp_results = run(
    (x0_batch, θ) -> lpg.ez_loss_and_grad_many(
        x0_batch,
        θ,
        solver(),
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
        #InterpolatingAdjoint(),
        #QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))
    ),
    init_policy_params, sample_x0
)

# Divide by two for the forward and backward passes.
 mean_euler_steps =
     mean((interp_results.nf_per_iter + interp_results.n∇f_per_iter) / batch_size / 2)
 euler_dt = T / mean_euler_steps
 @info euler_dt
#euler_dt = 0.05

@info "Euler"
euler_results = run(
    (x0_batch, θ) ->
        lpg.ez_euler_loss_and_grad_many(x0_batch, θ, euler_dt),
    init_policy_params, sample_x0
)

######## residual model
#=
@info "Residual Interp"
res_interp_results = run(
    (x0_batch, θ) -> rpg.ez_loss_and_grad_many(
        x0_batch,
        θ,
        solver(),
        InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
        #InterpolatingAdjoint(),
        #QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))
    ),
    res_policy_params, rsample_x0
)

# Divide by two for the forward and backward passes.
 mean_euler_steps =
     mean((res_interp_results.nf_per_iter + res_interp_results.n∇f_per_iter) / batch_size / 2)
 euler_dt = T / mean_euler_steps
#@info euler_dt
#euler_dt = 0.05

@info "Residual Euler"
res_euler_results = run(
    (x0_batch, θ) ->
        rpg.ez_euler_loss_and_grad_many(x0_batch, θ, euler_dt),
    res_policy_params, rsample_x0
)
=#


@info "Dumping results"
JLSO.save(
          "quadrotor_train_results.jlso",
          :interp_results => interp_results,
          :euler_results  => euler_results,
          :res_interp_results => res_interp_results,
          :res_euler_results  => res_euler_results,
         )

using UnicodePlots
xmax = maximum(vcat(sum(euler_results.nf_per_iter) +sum(euler_results.n∇f_per_iter),
                    sum(interp_results.nf_per_iter)+sum(interp_results.n∇f_per_iter),
                    sum(res_euler_results.nf_per_iter) +sum(res_euler_results.n∇f_per_iter),
                    sum(res_interp_results.nf_per_iter)+sum(res_interp_results.n∇f_per_iter)))

#xmax = mean([sum(euler_results.nf_per_iter)+sum(euler_results.n∇f_per_iter),
#             sum(interp_results.nf_per_iter)+sum(interp_results.n∇f_per_iter)])
ymin, ymax = round.(extrema([euler_results.loss_per_iter; interp_results.loss_per_iter;
                             res_euler_results.loss_per_iter; res_interp_results.loss_per_iter;]))

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
          name = "PPG (ours), normall",
          color = :red
         )
lineplot!(plt, cumsum(res_euler_results.nf_per_iter + res_euler_results.n∇f_per_iter),
          res_euler_results.loss_per_iter,
          name = "Euler BPTT, residual",
          color=:green)
lineplot!(plt, cumsum(res_interp_results.nf_per_iter + res_interp_results.n∇f_per_iter),
          res_interp_results.loss_per_iter,
          name = "PPG (ours), residual",
          color=:white)
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
          name = "PPG (ours), normall",
          color = :red
         )
lineplot!(plt, 1:num_iters,
          res_euler_results.loss_per_iter,
          name = "Euler BPTT, residual",
          color=:green)
lineplot!(plt, 1:num_iters,
          res_interp_results.loss_per_iter,
          name = "PPG (ours), residual",
          color=:white)
display(plt)


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
lineplot!(plt, 
          1:num_iters,
          cumsum(res_euler_results.nf_per_iter + res_euler_results.n∇f_per_iter),
          name = "Euler BPTT, residual",
          color=:green)
lineplot!(plt, 
          1:num_iters,
          cumsum(res_interp_results.nf_per_iter + res_interp_results.n∇f_per_iter),
          name = "PPG (ours), residual",
          color=:white)
display(plt)


#=
x0 = rsample_x0();
roll_iter = Array(rollout(x0[1:end-4], lpg.aug_dynamics!, interp_results.policy_params_per_iter[end,:]))[1:2,:]
roll_eulr = Array(rollout(x0[1:end-4], lpg.aug_dynamics!, euler_results.policy_params_per_iter[end,:]))[1:2,:]
rollriter = Array(rollout(x0, rpg.aug_dynamics!, res_interp_results.policy_params_per_iter[end,:]))[1:2,:]
rollreulr = Array(rollout(x0, rpg.aug_dynamics!, res_euler_results.policy_params_per_iter[end,:]))[1:2,:]

xmin, xmax = extrema(vcat(roll_iter[1,:], roll_eulr[1,:], rollriter[1,:], rollreulr[1,:]))
ymin, ymax = extrema(vcat(roll_iter[2,:], roll_eulr[2,:], rollriter[2,:], rollreulr[2,:]))

plt = scatterplot(roll_iter[1,:], roll_iter[2,:],
               name="interp rollout", xlabel="x", ylabel="y",
               xlim=(xmin, xmax),
               ylim=(ymin, ymax))
scatterplot!(plt, roll_eulr[1,:], roll_eulr[2,:], name="euler rollout")
scatterplot!(plt, rollriter[1,:], rollriter[2,:], name="iterp residual rollout")
scatterplot!(plt, rollreulr[1,:], rollreulr[2,:], name="euler residual rollout")
=#

