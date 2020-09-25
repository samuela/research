"""A bouncy ball with gravity. Goal is to reach a specific height at t = T."""

include("ppg_toi.jl")

import DifferentialEquations: Tsit5
import UnicodePlots: lineplot

gravity = -9.8
dynamics(state, u) = [state[2], gravity]
cost(state, u) = 0
policy(state, t, p) = 0
toi_affect(state, dt) = [-dt * state[2] - state[1], -state[2]]
terminal_cost(state) = (state[1] - 5) ^ 2

T = 2.0
goodies = ppg_toi_goodies(dynamics, cost, policy, TOIStuff((x) -> x[1], toi_affect, 1e-6), T)

x0 = [10.0, 0.0]
cost_per_iter = []
for iter in 1:1000
    sol, pb1 = goodies.loss_pullback(x0, zeros(1), Tsit5(), Dict())
    xT = sol.solutions[end].u[end][2:end]
    local cost, pb2 = Zygote.pullback(terminal_cost, xT)

    (g_xT, ) = pb2(1.0)
    local g = [0; g_xT]
    g = pb1(g, InterpolatingAdjoint()).g_z0
    g = [g[2], 0]

    global x0 -= 0.01 * g

    @show iter
    @show cost
    @show xT
    ts = 0:0.01:T
    # Don't forget the first is for the cost!
    lineplot(ts, [z[2] for z in sol.(ts)]) |> show
    println()

    push!(cost_per_iter, cost)
end

@assert cost_per_iter[end] <= 1e-6

lineplot((1:length(cost_per_iter)), convert(Array{Float64}, cost_per_iter)) |> show
