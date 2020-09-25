"""Test that we only register down-crossings. This one is a little tricky
because although the true solution goes above zero, the integrators are smart
enough to actually skip any concrete steps above zero. So we need to do this
testing the interpolated points thing."""

include("ppg_toi.jl")

g = 9.8
dynamics(state, u) = [state[2], -g]
cost(state, u) = 0
policy(state, t, p) = 0
toi_affect(state, dt) = [-dt * state[2] - state[1], -state[2]]

T = 1.0
goodies = ppg_toi_goodies(dynamics, cost, policy, TOIStuff([(x) -> x[1]], toi_affect, 1e-6), T)

x0 = -1
v0 = 5
parabola(t) = t * v0 - g / 2 * t^2 + x0
peak_time = v0 / g
@assert parabola(peak_time) > 0
land_time = (v0 + sqrt(v0^2 + 2 * g * x0)) / g
@assert land_time < T
sol, _ = goodies.loss_pullback([x0, v0], zeros(1), nothing, Dict())

fts = 0:0.01:T
# Don't forget the first is for the cost!
lineplot(ts, [z[2] for z in sol.(ts)]) |> show
lineplot(ts, [z[3] for z in sol.(ts)]) |> show

@assert length(sol.solutions) == 2
@assert sol(peak_time) ≈ [0, parabola(peak_time), 0]
