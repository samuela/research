import DifferentialEquations: Tsit5
import Flux
import Flux: ADAM
import Flux.Data: DataLoader
import DiffEqFlux:
    FastChain, FastDense, initial_params, sciml_train, ODEProblem, solve
import Random: seed!, randn
import Plots: plot
import Statistics: mean
import Zygote
using Optim: LBFGS, BFGS, Fminbox
import DiffEqSensitivity:
    InterpolatingAdjoint, BacksolveAdjoint, QuadratureAdjoint
import LinearAlgebra: I
import LineSearches

using BenchmarkTools

seed!(123)

floatT = Float32
x_dim = 2
T = 10.0
batch_size = 1
# num_hidden = 64
# policy = FastChain(
#     FastDense(x_dim, num_hidden, tanh),
#     # FastDense(num_hidden, num_hidden, tanh),
#     FastDense(num_hidden, x_dim),
#     # (x, _) -> 2 * x,
# )
policy = FastDense(x_dim, x_dim) # linear policy

dynamics, cost, sample_x0 = linear_env(floatT, x_dim, I, I, I, I)

function aug_dynamics!(dz, z, policy_params, t)
    x = @view z[2:end]
    u = policy(x, policy_params)
    dz[1] = cost(x, u)
    # Note that dynamics!(dz[2:end], x, u) breaks Zygote :(
    dz[2:end] = dynamics(x, u)
end

# function aug_dynamics(z, policy_params, t)
#     x = @view z[2:end]
#     u = policy(x, policy_params)
#     vcat(cost(x, u), dynamics(x, u))
# end

# function aug_dynamics2(z, policy_params, t)
#     x = z[2:end]
#     u = policy(x, policy_params)
#     [cost(x, u), dynamics(x, u)...]
# end

# @benchmark aug_dynamics!(
#     rand(floatT, x_dim + 1),
#     rand(floatT, x_dim + 1),
#     init_policy_params,
#     0.0,
# )

function loss(policy_params, data...)
    # TODO: use the ensemble thing
    mean([
        begin
            z0 = [0f0, x0...]
            rollout = solve(
                ODEProblem(aug_dynamics!, z0, (0, T), policy_params),
                Tsit5(),
                u0 = z0,
                p = policy_params,
                # sensealg = QuadratureAdjoint(),
            )
            Array(rollout)[1, end]
        end for x0 in data
    ])
end

callback = function (policy_params, loss_val)
    println("Loss $loss_val")
    false
end

# data = DataLoader([sample_x0() for _ = 1:1_000_000], batchsize = batch_size)

policy_params = initial_params(policy)
opt = ADAM()
# opt = BFGS(
#     alphaguess = LineSearches.InitialStatic(alpha = 0.1),
#     linesearch = LineSearches.Static(),
# )
for iter = 1:10
    @time begin
        println("start iter")
        # x0_batch = [sample_x0() for _ = 1:batch_size]
        x0 = sample_x0()
        @time loss_, vjp = Zygote.pullback(loss, policy_params, x0)
        @time g, _ = vjp(1)
        Flux.Optimise.update!(opt, policy_params, g)
        println("Episode $iter, loss = $loss_")
    end
end

# res1 = sciml_train(
#     loss,
#     init_policy_params,
#     BFGS(
#         alphaguess = LineSearches.InitialStatic(alpha = 0.1),
#         linesearch = LineSearches.Static(),
#     ),
#     # LBFGS(
#     #     alphaguess = LineSearches.InitialStatic(alpha = 0.01),
#     #     linesearch = LineSearches.Static(),
#     # ),
#     data,
#     cb = callback,
#     iterations = length(data),
#     allow_f_increases = true,
#     x_abstol = NaN,
#     x_reltol = NaN,
#     f_abstol = NaN,
#     f_reltol = NaN,
#     g_abstol = NaN,
#     g_reltol = NaN,
# )
