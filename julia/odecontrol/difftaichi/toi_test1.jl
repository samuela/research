"""The "billiards" ball example from the DiffTaichi paper."""

include("ppg_toi.jl")

import DifferentialEquations: Tsit5

dynamics(state, u) = [state[2], 0]
cost(state, u) = 0
policy(state, t, p) = 0
toi_affect(state, dt) = [-dt * state[2] - state[1], -state[2]]
terminal_cost(state) = (state[1] - 0.6) ^ 2

goodies = ppg_toi_goodies(dynamics, cost, policy, TOIStuff([(x) -> x[1]], toi_affect, 1e-3), 1.0)

x0 = [0.75, -1]
for _ in 1:100
    sol, pb1 = goodies.loss_pullback(x0, zeros(1), nothing, Dict())
    xT = sol.solutions[end].u[end][2:end]
    local cost, pb2 = Zygote.pullback(terminal_cost, xT)

    (g_xT, ) = pb2(1.0)
    local g = [0; g_xT]
    g = pb1(g, InterpolatingAdjoint()).g_z0
    g = [g[2], 0]

    global x0 -= 0.1 * g
end
@assert x0 ≈ [0.4, -1.0]
