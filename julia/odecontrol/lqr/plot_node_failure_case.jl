import DifferentialEquations: Tsit5
import Flux
import Flux: ADAM
import DiffEqFlux: FastChain, FastDense, initial_params, ODEProblem, solve
import Random: seed!
import Statistics: mean
import Zygote
import DiffEqSensitivity:
    InterpolatingAdjoint, BacksolveAdjoint, ODEAdjointProblem
import LinearAlgebra: I, norm
import ControlSystems
import PyPlot

include("common.jl")
include("../utils.jl")

seed!(123)

floatT = Float32
x_dim = 2
T = 25.0
num_iters = 1000
x0 = ones(x_dim)

num_hidden = 32
policy = FastChain(
    FastDense(x_dim, num_hidden, tanh),
    # FastDense(num_hidden, num_hidden, tanh),
    FastDense(num_hidden, x_dim),
)
init_policy_params = initial_params(policy) * 0.1

const A = Matrix{floatT}(0 * I, x_dim, x_dim)
const B = Matrix{floatT}(I, x_dim, x_dim)
const Q = Matrix{floatT}(I, x_dim, x_dim)
const R = Matrix{floatT}(I, x_dim, x_dim)
const K = ControlSystems.lqr(A, B, Q, R)
dynamics, cost, sample_x0 = LinearEnv.linear_env(floatT, x_dim, 0 * I, I, I, I)

lqr_policy_loss =
    policy_loss(dynamics, cost, (x, _) -> -K * x, InterpolatingAdjoint())
lqr_loss = lqr_policy_loss(nothing, [x0])

function aug_dynamics!(dz, z, policy_params, t)
    x = @view z[2:end]
    u = policy(x, policy_params)
    dz[1] = cost(x, u)
    # Note that dynamics!(dz[2:end], x, u) breaks Zygote :(
    dz[2:end] = dynamics(x, u)
end

"""Calculate the loss, gradient wrt parameters, and the reconstructed z(0)."""
function node_loss_and_grad(policy_params, x0)
    z0 = vcat(0.0, x0)
    z_dim = x_dim + 1
    fwd_sol = solve(
        ODEProblem(aug_dynamics!, z0, (0, T), policy_params),
        Tsit5(),
        u0 = z0,
        p = policy_params,
    )
    # See https://diffeq.sciml.ai/stable/analysis/sensitivity/#Syntax-1.
    bwd_sol = solve(
        ODEAdjointProblem(
            fwd_sol,
            BacksolveAdjoint(checkpointing = false),
            # This algebra is annoying but this is where BacksolveAdjoint
            # happens to put the first element of z(t).
            (out, x, p, t, i) -> (fill!(out, 0); out[end-x_dim] = 1),
            [T],
        ),
        Tsit5(),
        dense = false,
        save_everystep = false,
    )

    # The first z_dim elements of bwd_sol.u are the gradient wrt z0, next
    # however many are the gradient wrt policy_params, final z_dim are the
    # reconstructed z(t) trajectory. Why the gradients come out negative is one
    # of the great mysteries of our time...
    (
        fwd_sol.u[end][1],
        -bwd_sol.u[end][z_dim+1:end-z_dim],
        bwd_sol.u[end][end-z_dim+1:end],
    )
end

@info "Neural ODE"
function plot_neural_ode()
    policy_params = deepcopy(init_policy_params)
    learned_loss_per_iter = fill(NaN, num_iters)
    reconst_error_per_iter = fill(NaN, num_iters)
    opt = ADAM()
    for iter = 1:num_iters
        @time begin
            loss_, g, z0_reconst = node_loss_and_grad(policy_params, x0)
            Flux.Optimise.update!(opt, policy_params, g)
            learned_loss_per_iter[iter] = loss_
            reconst_error_per_iter[iter] = norm(z0_reconst - vcat(0.0, x0))
            # println("Episode $iter, excess loss = $(loss_ - lqr_loss)")
        end
    end

    begin
        _, ax1 = PyPlot.subplots()
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss \$\\mathcal{L}(x, \\theta)\$", color = "tab:blue")
        ax1.tick_params(axis = "y", labelcolor = "tab:blue")
        # ax1.set_yscale("log")
        ax1.plot(learned_loss_per_iter, color = "tab:blue")
        PyPlot.axhline(lqr_loss, linestyle = "--", color = "grey")

        ax2 = ax1.twinx()
        ax2.set_ylabel("L2 error", color = "tab:red")
        ax2.tick_params(axis = "y", labelcolor = "tab:red")
        ax2.plot([], color = "tab:blue", label = "Learned policy loss")
        ax2.plot([], linestyle = "--", color = "grey", label = "LQR solution")
        ax2.plot(
            reconst_error_per_iter,
            color = "tab:red",
            label = "Backsolve error",
        )

        PyPlot.legend()
        PyPlot.tight_layout()
        PyPlot.savefig("node_comparison.pdf")
    end
end

################################################################################
@info "Interpolation adjoint"
function plot_interp()
    learned_policy_loss =
        policy_loss(dynamics, cost, policy, InterpolatingAdjoint())

    policy_params = deepcopy(init_policy_params)
    learned_loss_per_iter = fill(NaN, num_iters)
    opt = ADAM()
    for iter = 1:num_iters
        @time begin
            loss_, vjp =
                Zygote.pullback(learned_policy_loss, policy_params, [x0])
            g, _ = vjp(1)
            Flux.Optimise.update!(opt, policy_params, g)
            learned_loss_per_iter[iter] = loss_
            # println("Episode $iter, excess loss = $(loss_ - lqr_loss)")
        end
    end

    begin
        _, ax1 = PyPlot.subplots()
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss \$\\mathcal{L}(x, \\theta)\$", color = "tab:blue")
        ax1.tick_params(axis = "y", labelcolor = "tab:blue")
        # ax1.set_yscale("log")
        ax1.plot(learned_loss_per_iter, color = "tab:blue")
        PyPlot.axhline(lqr_loss, linestyle = "--", color = "grey")

        ax2 = ax1.twinx()
        ax2.set_ylabel("L2 error", color = "tab:red")
        ax2.tick_params(axis = "y", labelcolor = "tab:red")
        ax2.plot([], color = "tab:blue", label = "Learned policy loss")
        ax2.plot([], linestyle = "--", color = "grey", label = "LQR solution")

        # Interpolated forward pass has no reconstruction error since we have a
        # knot point at x(0)!
        ax2.plot(
            zeros(size(learned_loss_per_iter)),
            color = "tab:red",
            label = "Backsolve error",
        )
        ax2.set_ylim(-1.0, 50.0)

        PyPlot.legend()
        PyPlot.tight_layout()
        PyPlot.savefig("ours_comparison.pdf")
    end
end

plot_neural_ode()
plot_interp()
