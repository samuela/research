"""Test gradients on init conditions, no contacts."""

include("ppg_toi.jl")

import DifferentialEquations: Tsit5
import UnicodePlots: lineplot

v_dynamics(v, x, u) = [0.0]
x_dynamics(v, x, u) = v
cost(v, x, u) = 0.0
policy(v, x, p, t) = 0.0
terminal_cost(x) = (x[1] - 0.6) ^ 2

T = 1.0
loss_pullback = ppg_toi_goodies(
    v_dynamics,
    x_dynamics,
    cost,
    policy,
    TOIStuff([], (x, v, dt) -> begin @error "this should never happen" end, 1e-3),
    T
)

v0 = [-1.0]
x0 = [0.75]

sol, _ = loss_pullback(v0, x0, zeros(), nothing, Dict())

ts = 0:0.01:T
lineplot(ts, [z.x[2][2] for z in sol.(ts)], title = "x") |> show
lineplot(ts, [z.x[1][2] for z in sol.(ts)], title = "v") |> show

@assert length(sol.solutions) == 1

for _ in 1:100
    local sol, pb1 = loss_pullback(v0, x0, zeros(), nothing, Dict())
    xT = sol.solutions[end].u[end].x[2][2:end]
    local loss, pb2 = Zygote.pullback(terminal_cost, xT)

    (g_xT, ) = pb2(1.0)
    local g = ([0.0; zero(v0)], [0.0; g_xT])
    g = pb1(g, InterpolatingAdjoint()).g_z0
    g = g[2][2:end]

    global x0 -= 0.1 * g

    # @show loss
    # @show sum(g .^ 2)
end
@assert x0 ≈ [1.6]

sol, pb1 = loss_pullback(v0, x0, zeros(), nothing, Dict())
xT = sol.solutions[end].u[end].x[2][2:end]
loss, pb2 = Zygote.pullback(terminal_cost, xT)

(g_xT, ) = pb2(1.0)
g = ([0.0; zero(v0)], [0.0; g_xT])
g = pb1(g, InterpolatingAdjoint()).g_z0
g = g[2][2:end]

@assert loss < eps()
@assert sum(g .^ 2) < eps()
