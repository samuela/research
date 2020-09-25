"""Test that we can handle multiple contacts with bouncy balls and gradients."""

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
    tois = [-old_x[i] / min(old_v[i], -eps()) for i in 1:n_objects]
    impact_velocities = -old_v

    # Can't use zip. See https://github.com/FluxML/Zygote.jl/issues/221.
    new_v = [if tois[i] < dt impact_velocities[i] else old_v[i] end for i in 1:n_objects]
    new_x = old_x + min.(tois, dt) .* old_v + max.(dt .- tois, 0) .* new_v
    [new_x; new_v]
end

T = 1.23
goodies = ppg_toi_goodies(dynamics, cost, policy, TOIStuff([(x) -> x[i] for i in 1:n_objects], toi_affect, 1e-3), T)

x0 = [0.25, 0.5, -1, -1]
sol, pullback = goodies.loss_pullback(x0, zeros(1), nothing, Dict())
@assert length(sol.solutions) == 3

pb = pullback([0, 1, 0, 0, 0], InterpolatingAdjoint())
@assert pb.g_z0 ≈ [0, -1, 0, -T, 0]
pb = pullback([0, 0, 1, 0, 0], InterpolatingAdjoint())
@assert pb.g_z0 ≈ [0, 0, -1, 0, -T]

ts = 0:0.01:T
# Don't forget the first is for the cost!
lineplot(ts, [z[2] for z in sol.(ts)]) |> show
lineplot(ts, [z[3] for z in sol.(ts)]) |> show
