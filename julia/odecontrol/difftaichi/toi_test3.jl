"""Test that we can handle multiple contacts."""

include("ppg_toi.jl")

import DifferentialEquations
import UnicodePlots: lineplot

n_objects = 2
dynamics(state, u) = [state[3], state[4], 0, 0]
cost(state, u) = 0
policy(state, t, p) = 0
toi_affect(state, dt) = begin
    old_x, old_v = state[1:2], state[3:4]
    # We want to make sure that we only register for negative velocities.
    tois = [-x / min(v, -eps()) for (x, v) in zip(old_x, old_v)]
    impact_velocity = 0 # [0, 0] for mass_spring

    new_v = [if toi < dt impact_velocity else v end for (toi, v) in zip(tois, old_v)]
    new_x = old_x + min.(tois, dt) .* old_v + max.(dt .- tois, 0) .* new_v
    [new_x; new_v]
end
terminal_cost(state) = (state[1] - 0.6) ^ 2

T = 1.0
goodies = ppg_toi_goodies(dynamics, cost, policy, TOIStuff([(x) -> x[i] for i in 1:n_objects], toi_affect, 1e-3), T)

x0 = [0.25, 0.5, -1, -1]
sol, _ = goodies.loss_pullback(x0, zeros(1), nothing, Dict())

@assert length(sol.solutions) == 3

ts = 0:0.01:T
# Don't forget the first is for the cost!
lineplot(ts, [z[2] for z in sol.(ts)]) |> show
lineplot(ts, [z[3] for z in sol.(ts)]) |> show
