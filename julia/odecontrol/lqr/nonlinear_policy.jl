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
import ControlSystems

include("common.jl")
include("../utils.jl")

seed!(123)

floatT = Float32
x_dim = 2
T = 25.0
num_hidden = 32
policy = FastChain(
    FastDense(x_dim, num_hidden, tanh),
    # FastDense(num_hidden, num_hidden, tanh),
    FastDense(num_hidden, x_dim),
)
# policy = FastDense(x_dim, x_dim) # linear policy

const A = Matrix{floatT}(0 * I, x_dim, x_dim)
const B = Matrix{floatT}(I, x_dim, x_dim)
const Q = Matrix{floatT}(I, x_dim, x_dim)
const R = Matrix{floatT}(I, x_dim, x_dim)
dynamics, cost, sample_x0 = LinearEnv.linear_env(floatT, x_dim, 0 * I, I, I, I)

const K = ControlSystems.lqr(A, B, Q, R)

learned_policy_loss =
    policy_loss(dynamics, cost, policy, InterpolatingAdjoint())
lqr_policy_loss =
    policy_loss(dynamics, cost, (x, _) -> -K * x, InterpolatingAdjoint())

# @info "Calculating LQR loss"
# @time lqr_loss = loss(lqr_params, [sample_x0() for _ = 1:1024])

@info "Training policy"
num_iters = 1000
batch_size = 1
learned_loss_per_iter = fill(NaN, num_iters)
lqr_loss_per_iter = fill(NaN, num_iters)
policy_params = initial_params(policy) * 0.1
opt = ADAM()
# opt = BFGS(
#     alphaguess = LineSearches.InitialStatic(alpha = 0.1),
#     linesearch = LineSearches.Static(),
# )
for iter = 1:num_iters
    @time begin
        # x0_batch = [sample_x0() for _ = 1:batch_size]
        x0_batch = [ones(x_dim)]
        loss_, vjp =
            Zygote.pullback(learned_policy_loss, policy_params, x0_batch)
        lqr_loss = lqr_policy_loss(nothing, x0_batch)
        g, _ = vjp(1)
        Flux.Optimise.update!(opt, policy_params, g)
        learned_loss_per_iter[iter] = loss_
        lqr_loss_per_iter[iter] = lqr_loss
        println("Episode $iter, excess loss = $(loss_ - lqr_loss)")
    end
end

import Plots: plot
plot(learned_loss_per_iter - lqr_loss_per_iter)

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
