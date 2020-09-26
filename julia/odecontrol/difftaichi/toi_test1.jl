"""The "billiards" ball example from the DiffTaichi paper."""

include("ppg_toi.jl")

import DifferentialEquations: Tsit5
import UnicodePlots: lineplot

v_dynamics(v, x, u) = [0.0]
x_dynamics(v, x, u) = v
cost(v, x, u) = 0.0
policy(v, x, p, t) = 0.0
toi_affect(v, x, dt) = (-v, -dt * v - x)
terminal_cost(state) = (state[1] - 0.6) ^ 2

T = 1.0
# goodies = ppg_toi_goodies(dynamics, cost, policy, TOIStuff([(x) -> x[1]], toi_affect, 1e-3), 1.0)
loss_pullback = ppg_toi_goodies(
    v_dynamics,
    x_dynamics,
    cost,
    policy,
    TOIStuff([(v, x) -> x[1]], toi_affect, 1e-3),
    # TOIStuff([], toi_affect, 1e-3),
    T)

v0 = [-1.0]
x0 = [0.75]

sol, _ = loss_pullback(v0, x0, zeros(), nothing, Dict())

ts = 0:0.01:T
lineplot(ts, [z.x[2].x[2][1] for z in sol.(ts)], title = "x") |> show
lineplot(ts, [z.x[1].x[2][1] for z in sol.(ts)], title = "v") |> show

@assert length(sol.solutions) == 2

for _ in 1:1
    local sol, pb1 = loss_pullback(v0, x0, zeros(), nothing, Dict())
    xT = sol.solutions[end].u[end].x[2].x[2]
    local cost, pb2 = Zygote.pullback(terminal_cost, xT)

    (g_xT, ) = pb2(1.0)
    local g = ArrayPartition(ArrayPartition(zeros(), zero(v0)), ArrayPartition(zeros(), g_xT))
    g = pb1(g, InterpolatingAdjoint()).g_z0
    @error "yay"
    g = [g[2], 0]

    global x0 -= 0.1 * g
end
# @assert x0 ≈ [0.4, -1.0]
