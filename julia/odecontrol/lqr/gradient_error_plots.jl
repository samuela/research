import DifferentialEquations: Tsit5, Vern9
import DiffEqFlux: FastChain, FastDense, initial_params, ODEProblem, solve
import Random: seed!
import Statistics: mean, std, median
import Zygote
import DiffEqSensitivity:
    InterpolatingAdjoint, BacksolveAdjoint, QuadratureAdjoint, ODEAdjointProblem
import LinearAlgebra: I, norm
import Plots
import PyPlot
import ControlSystems

include("common.jl")

seed!(123)

floatT = Float32
x_dim = 2
T = 10.0
num_samples = 10

# num_hidden = 64
# policy = FastChain(
#     FastDense(x_dim, num_hidden, tanh),
#     # FastDense(num_hidden, num_hidden, tanh),
#     FastDense(num_hidden, x_dim),
#     # (x, _) -> 2 * x,
# )

# A linear policy. Use this when operating with lqr_params below.
policy = FastDense(x_dim, x_dim)

# A = Matrix{floatT}(0 * I, x_dim, x_dim)
# B = Matrix{floatT}(I, x_dim, x_dim)
# Q = Matrix{floatT}(I, x_dim, x_dim)
# R = Matrix{floatT}(I, x_dim, x_dim)

const A = 0 * I
const B = I
const Q = I
const R = I
dynamics, cost, sample_x0 = LinearEnv.linear_env(floatT, x_dim, A, B, Q, R)

# lqr needs to be able to infer the dimension. See https://stackoverflow.com/questions/57270276/identity-matrix-in-julia.
const K = ControlSystems.lqr(Matrix{Float64}(A, x_dim, x_dim), B, Q, R)
# We need to add zeros at the end for the bias term. See also https://github.com/SciML/DiffEqFlux.jl/blob/fa5d1678def49b0f0f3bad99e646f78c9a2a9d71/src/fast_layers.jl#L43.
const lqr_params = vcat(convert(Array{floatT}, -K[:]), zeros(floatT, x_dim))

function policy_dynamics!(dx, x, policy_params, t)
    u = policy(x, policy_params)
    dx .= dynamics(x, u)
end

function cost_functional(x, policy_params, t)
    cost(x, policy(x, policy_params))
end

# See https://github.com/SciML/DiffEqSensitivity.jl/issues/302 for context.
dcost_tuple = (
    (out, u, p, t) -> begin
        ū, _, _ = Zygote.gradient(cost_functional, u, p, t)
        out .= ū
    end,
    (out, u, p, t) -> begin
        _, p̄, _ = Zygote.gradient(cost_functional, u, p, t)
        out .= p̄
    end,
)

function gold_standard_gradient(x0, policy_params)
    # Actual/gold standard evaluation. Using high-fidelity Vern9 method with
    # small tolerances. We want to use Float64s for maximum accuracy. Also 1e-14
    # is recommended as the minimum allowable tolerance here: https://docs.sciml.ai/stable/basics/faq/#How-to-get-to-zero-error-1.
    x0_f64 = convert(Array{Float64}, x0)
    policy_params_f64 = convert(Array{Float64}, policy_params)
    fwd_sol = solve(
        ODEProblem(policy_dynamics!, x0, (0, T), policy_params),
        Vern9(),
        u0 = x0_f64,
        p = policy_params_f64,
        abstol = 1e-14,
        reltol = 1e-14,
    )
    # Note that specifying dense = false is essential for getting acceptable
    # performance. save_everystep = false is another small win.
    bwd_sol = solve(
        ODEAdjointProblem(
            fwd_sol,
            InterpolatingAdjoint(),
            cost_functional,
            nothing,
            dcost_tuple,
        ),
        Vern9(),
        dense = false,
        save_everystep = false,
        abstol = 1e-14,
        reltol = 1e-14,
    )
    @assert typeof(fwd_sol.u) == Array{Array{Float64,1},1}
    @assert typeof(bwd_sol.u) == Array{Array{Float64,1},1}

    # Note that the backwards solution includes the gradient on x0, as well as
    # policy_params.
    (fwd = fwd_sol, bwd = bwd_sol)
end

function eval_interp(x0, policy_params, abstol, reltol)
    fwd_sol = solve(
        ODEProblem(policy_dynamics!, x0, (0, T), policy_params),
        Tsit5(),
        u0 = x0,
        p = policy_params,
        abstol = abstol,
        reltol = reltol,
    )
    bwd_sol = solve(
        ODEAdjointProblem(
            fwd_sol,
            InterpolatingAdjoint(),
            cost_functional,
            nothing,
            dcost_tuple,
        ),
        Tsit5(),
        dense = false,
        save_everystep = false,
        abstol = abstol,
        reltol = reltol,
    )
    @assert typeof(fwd_sol.u) == Array{Array{floatT,1},1}
    @assert typeof(bwd_sol.u) == Array{Array{floatT,1},1}

    # Note that g includes the x0 gradient and the gradient on parameters.
    # We do exactly as many f calls as there are function calls in the forward
    # pass, and in the backward pass we don't need to call f, but instead we
    # call ∇f.
    (
        fwd = fwd_sol,
        bwd = bwd_sol,
        g = bwd_sol.u[end],
        nf = fwd_sol.destats.nf,
        n∇f = bwd_sol.destats.nf,
    )
end

function eval_backsolve(x0, policy_params, abstol, reltol, checkpointing)
    fwd_sol = solve(
        ODEProblem(policy_dynamics!, x0, (0, T), policy_params),
        Tsit5(),
        u0 = x0,
        p = policy_params,
        abstol = abstol,
        reltol = reltol,
    )
    bwd_sol = solve(
        ODEAdjointProblem(
            fwd_sol,
            BacksolveAdjoint(checkpointing = checkpointing),
            cost_functional,
            nothing,
            dcost_tuple,
        ),
        Tsit5(),
        dense = false,
        save_everystep = false,
        abstol = abstol,
        reltol = reltol,
    )
    @assert typeof(fwd_sol.u) == Array{Array{floatT,1},1}
    @assert typeof(bwd_sol.u) == Array{Array{floatT,1},1}

    # In the backsolve adjoint, the last x_dim dimensions are for the
    # reconstructed x state.
    # When running the backsolve adjoint we have additional f evaluations every
    # step of the backwards pass, since we need -f to reconstruct the x path.
    (
        fwd = fwd_sol,
        bwd = bwd_sol,
        g = bwd_sol.u[end][1:end-x_dim],
        nf = fwd_sol.destats.nf + bwd_sol.destats.nf,
        n∇f = bwd_sol.destats.nf,
    )
end

# See https://github.com/SciML/DiffEqSensitivity.jl/issues/304 for why this
# doesn't work.
# function eval_quadrature(x0, policy_params, abstol, reltol)
#     fwd_sol = solve(
#         ODEProblem(policy_dynamics!, x0, (0, T), policy_params),
#         Tsit5(),
#         u0 = x0,
#         p = policy_params,
#         abstol = abstol,
#         reltol = reltol,
#     )
#     bwd_sol = solve(
#         ODEAdjointProblem(fwd_sol, QuadratureAdjoint(), cost_functional),
#         Tsit5(),
#         # dense = false,
#         # save_everystep = false,
#         abstol = abstol,
#         reltol = reltol,
#     )
#     @assert typeof(fwd_sol.u) == Array{Array{floatT,1},1}
#     @assert typeof(bwd_sol.u) == Array{Array{floatT,1},1}
#
#     # The way to do this is defined here: https://github.com/SciML/DiffEqSensitivity.jl/blob/master/src/local_sensitivity/quadrature_adjoint.jl#L173
#
#     # TODO: figure out how to get the number of function evals from quadgk...
#
#     # (fwd = fwd_sol, bwd = estbwd_sol, g = estbwd_sol.u[end][1:end-x_dim])
# end

function plot_results!(results, label)
    nf_calls = [[sol.nf + sol.n∇f for sol in res] for res in results]
    g_errors = [
        [
            norm(gold.bwd.u[end] - est.g)
            for (gold, est) in zip(gold_standard_results, res)
        ] for res in results
    ]

    function safe_error_bars(vs)
        vs_median = map(median, vs)
        vs_mean = map(mean, vs)
        vs_std = map(std, vs)
        collect(zip(
            [
                min(ṽ - 1e-6, σ + ṽ - μ)
                for (ṽ, μ, σ) in zip(vs_median, vs_mean, vs_std)
            ],
            vs_std + vs_mean - vs_median,
        ))
    end

    Plots.plot!(
        map(median, nf_calls),
        map(median, g_errors),
        xlabel = "Function evaluations",
        ylabel = "L2 error in the gradient",
        xaxis = :log10,
        yaxis = :log10,
        xerror = safe_error_bars(nf_calls),
        # yerror = safe_error_bars(g_errors),
        label = label,
    )
end

# Absolute error tolerances should generally be smaller than relative
# tolerances.
reltols = 0.1 .^ (0:9)
abstols = 1e-3 * reltols

# init_conditions = [(sample_x0(), initial_params(policy)) for _ = 1:num_samples]
init_conditions = [(sample_x0(), lqr_params) for _ = 1:num_samples]

# These things are really, really slow on my computer
@info "gold standard"
@time gold_standard_results =
    [gold_standard_gradient(x0, θ) for (x0, θ) in init_conditions]

@info "interp"
@time interp_results = [
    [eval_interp(x0, θ, atol, rtol) for (x0, θ) in init_conditions] for (atol, rtol) in zip(abstols, reltols)
]
@info "backsolve"
@time backsolve_results = [
    [eval_backsolve(x0, θ, atol, rtol, false) for (x0, θ) in init_conditions] for (atol, rtol) in zip(abstols, reltols)
]
@info "backsolve with checkpoints"
@time backsolve_checkpointing_results = [
    [eval_backsolve(x0, θ, atol, rtol, true) for (x0, θ) in init_conditions] for (atol, rtol) in zip(abstols, reltols)
]

# The rest is useless...
# quadrature_results = [
#     [eval_quadrature(x0, θ, atol, rtol) for (x0, θ) in init_conditions] for (atol, rtol) in zip(abstols, reltols)
# ]

# Plots.pyplot()
# Plots.PyPlotBackend()
# pgfplotsx()

# Doing plot() clears the current figure.
# Plots.plot(title = "Compute/accuracy tradeoff")
# plot_results!(backsolve_results, "Neural ODE")
# plot_results!(backsolve_checkpointing_results, "Neural ODE with checkpointing")
# plot_results!(interp_results, "Ours")
