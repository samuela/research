import DifferentialEquations: Tsit5
import Flux
import Flux: ADAM
import Flux.Data: DataLoader
import DiffEqFlux:
    FastChain, FastDense, initial_params, sciml_train, ODEProblem, solve
import Random: seed!, randn
import Plots
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
T = 5.0

dynamics, cost, sample_x0, obs =
    DiffDriveEnv.diffdrive_env(floatT, 1.0f0, 0.5f0)

num_hidden = 32
policy = FastChain(
    (x, _) -> obs(x),
    FastDense(7, num_hidden, tanh),
    FastDense(num_hidden, num_hidden, tanh),
    FastDense(num_hidden, 2),
)
# policy = FastDense(x_dim, x_dim) # linear policy

learned_policy_loss =
    policy_loss(dynamics, cost, policy, InterpolatingAdjoint())

@info "Training policy"
num_iters = 25
learned_loss_per_iter = fill(NaN, num_iters)
x0_test_batch = [sample_x0() for _ = 1:10]
policy_params = initial_params(policy) * 0.1
opt = ADAM()
anim = Plots.Animation()
for iter = 1:num_iters
    @time begin
        x0_batch = x0_test_batch
        loss_, vjp =
            Zygote.pullback(learned_policy_loss, policy_params, x0_batch)
        g, _ = vjp(1)
        Flux.Optimise.update!(opt, policy_params, g)
        learned_loss_per_iter[iter] = loss_
        println("Episode $iter, loss = $(loss_)")
    end

    begin
        trajs = [
            begin
                sol = solve(
                    ODEProblem(
                        (x, p, t) -> dynamics(x, policy(x, policy_params)),
                        x0,
                        (0, T),
                        policy_params,
                    ),
                    Tsit5(),
                )
                traj = sol.(0:0.1:T)
                xs = [s[1] for s in traj]
                ys = [s[2] for s in traj]
                (xs, ys)
            end for x0 in x0_test_batch
        ]

        p = Plots.plot(
            title = "Iteration $iter",
            legend = false,
            aspect_ratio = :equal,
            xlims = (-7.5, 7.5),
            ylims = (-7.5, 7.5),
        )
        for (xs, ys) in trajs
            Plots.plot!(xs, ys)
            Plots.scatter!([xs[1]], [ys[1]], color = :grey, markersize = 5)
        end
        Plots.frame(anim)
    end
end
Plots.gif(anim, "diffdrive.gif")
