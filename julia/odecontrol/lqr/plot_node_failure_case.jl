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

lqr_goodies =
    ppg_goodies(dynamics, cost, (x, _) -> -K * x)
lqr_sol, _ = lqr_goodies.loss_pullback(x0, nothing, nothing)
lqr_loss = lqr_sol[end][1]

learned_policy_goodies = ppg_goodies(dynamics, cost, policy)

"""Calculate the loss, gradient wrt parameters, and the reconstructed z(0)."""
function node_loss_and_grad(x0, policy_params)
    z_dim = x_dim + 1
    fwd_sol, vjp = learned_policy_goodies.loss_pullback(x0, policy_params, BacksolveAdjoint(checkpointing = false))
    bwd_sol = vjp(vcat(1, zero(x0)))
    loss, _ = extract_loss_and_xT(fwd_sol)
    _, g = extract_gradients(fwd_sol, bwd_sol)

    # The final z_dim elements of bwd_sol are the reconstructed z(t) trajectory.
    loss, g, bwd_sol[end][end-z_dim+1:end]
end

function plot_neural_ode()
    policy_params = deepcopy(init_policy_params)
    learned_loss_per_iter = fill(NaN, num_iters)
    reconst_error_per_iter = fill(NaN, num_iters)
    opt = ADAM()
    for iter = 1:num_iters
        @time begin
            loss, g, z0_reconst = node_loss_and_grad(x0, policy_params)
            Flux.Optimise.update!(opt, policy_params, g)
            learned_loss_per_iter[iter] = loss
            reconst_error_per_iter[iter] = norm(z0_reconst - vcat(0.0, x0))
            # println("Episode $iter, excess loss = $(loss - lqr_loss)")
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
function plot_interp()
    policy_params = deepcopy(init_policy_params)
    learned_loss_per_iter = fill(NaN, num_iters)
    opt = ADAM()
    for iter = 1:num_iters
        @time begin
            fwd_sol, vjp = learned_policy_goodies.loss_pullback(x0, policy_params, InterpolatingAdjoint())
            bwd_sol = vjp(vcat(1, zero(x0)))
            loss, _ = extract_loss_and_xT(fwd_sol)
            _, g = extract_gradients(fwd_sol, bwd_sol)

            Flux.Optimise.update!(opt, policy_params, g)
            learned_loss_per_iter[iter] = loss
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

@info "Neural ODE"
plot_neural_ode()
@info "Interpolation adjoint"
plot_interp()
