include("ppg_toi.jl")

import DiffEqFlux: FastChain, FastDense, initial_params
import Random: seed!
import Flux
import Flux: Momentum
import ProgressMeter
import DifferentialEquations: Tsit5
import LinearAlgebra: norm
import DiffEqSensitivity: ZygoteVJP
import PyCall: @pyimport, @py_str, pyimport

# See https://github.com/JuliaPy/PyCall.jl/issues/48#issuecomment-515787405.
py"""
import sys
sys.path.insert(0, "./difftaichi/python")
"""

importlib = pyimport("importlib")
electric = pyimport("electric")
importlib.reload(electric)

# DiffTaichi takes 512 steps each of length dt = 0.03.
T = 15.36

# DiffTaichi uses 512 // 256 = 2.
num_segments = 2

pad = 0.1
gravitation_position = [[pad, pad], [pad, 1 - pad], [1 - pad, 1 - pad],
                        [1 - pad, pad], [0.5, 1 - pad], [0.5, pad], [pad, 0.5],
                        [1 - pad, 0.5]]

# The exact equivalent from the difftaichi code is that our dampening = (1 - exp(-dt * d)) / d where d is their damping.
# In difftaichi d = 0.2 and dt = 0.03 which comes out to...
damping = 0.029910179730323616
@assert damping >= 0

K = 1e-3

function sample_task()
    goal_points = rand(num_segments + 1, 2)
    # We nudge the final time step to be just a liiiiiitle bit bigger than T so that we don't overflow goal_points when
    # t = T.
    timesteps = range(0, nextfloat(T), length = num_segments + 1)

    goal_stuff(t) = begin
        ix = searchsortedlast(timesteps, t)
        a = goal_points[ix, :]
        b = goal_points[ix + 1, :]
        tt = (t - timesteps[ix]) / (timesteps[ix + 1] - timesteps[ix])
        goal = (1 - tt) * a + tt * b
        # Note that this isn't technically right since it doesn't account for time/distance. But this is how difftaichi
        # does it: https://github.com/yuanming-hu/difftaichi/blob/0ac795a5a4dafab50c52592448d224e71ee0328d/examples/electric.py#L170-L173.
        goal_v = b - a
        (goal, goal_v)
    end

    v_dynamics(v, x, u) = begin
        sum([begin
            r = x - gravitation_position[i]
            len_r = max(norm(r), 1e-1)
            K * u[i] / (len_r ^ 3) * r
        end for i in 1:length(gravitation_position)]) - damping * v
    end
    x_dynamics(v, x, u) = v
    cost(v, x, u, t) = begin
        goal, _ = goal_stuff(t)
        sum((x - goal) .^ 2)
    end
    observation(v, x, t) = begin
        goal, goal_v = goal_stuff(t)
        [x; v; goal; goal_v]
    end

    v0 = zeros(2)
    x0 = goal_points[1, :]
    v0, x0, v_dynamics, x_dynamics, cost, observation, goal_stuff
end

# DiffTaichi uses a single hidden layer with 64 units.
num_hidden = 64
policy = FastChain(
    FastDense(8, num_hidden, tanh),
    FastDense(num_hidden, size(gravitation_position, 1)),
)
init_policy_params = 0.1 * initial_params(policy)

function run_ppg(num_iters)
    # Seed here so that both interp and euler get the same batches.
    seed!(123)

    loss_per_iter = fill(NaN, num_iters)
    policy_params_per_iter = fill(NaN, num_iters, length(init_policy_params))
    g_per_iter = fill(NaN, num_iters, length(init_policy_params))
    nf_per_iter = fill(NaN, num_iters)
    n∇ₓf_per_iter = fill(NaN, num_iters)
    n∇ᵤf_per_iter = fill(NaN, num_iters)

    policy_params = deepcopy(init_policy_params)
    opt = Momentum(0.001)
    progress = ProgressMeter.Progress(num_iters)
    for iter in 1:num_iters
        # We have to sample a new task each time because we randomize where the goal goes.
        v0, x0, v_dyn, x_dyn, cost, observation, goal_stuff = sample_task()
        # Use the TOI version because it supports DynamicalODEProblems, just give it a no-op toi setup.
        sol, pullback = ppg_toi_goodies(
            v_dyn,
            x_dyn,
            cost,
            (v, x, params, t) -> policy(observation(v, x, t), params),
            TOIStuff([], (v, x, dt) -> begin @error "this should never happen" end, 1e-3),
            T
        )(v0, x0, policy_params, Tsit5(), Dict(:rtol => 1e-3, :atol => 1e-3))
        loss = sol.solutions[end].u[end].x[2][1]

        pb_stuff = pullback(([0.0; zeros(length(v0))], [1.0; zeros(length(x0))]), InterpolatingAdjoint(autojacvec = ZygoteVJP()))
        g = pb_stuff.g_p

        loss_per_iter[iter] = loss
        policy_params_per_iter[iter, :] = policy_params
        g_per_iter[iter, :] = g
        nf_per_iter[iter] = pb_stuff.nf + sum(s.destats.nf for s in sol.solutions)
        n∇ₓf_per_iter[iter] = pb_stuff.n∇ₓf
        n∇ᵤf_per_iter[iter] = pb_stuff.n∇ᵤf

        clamp!(g, -10, 10)
        Flux.Optimise.update!(opt, policy_params, g)
        ProgressMeter.next!(
            progress;
            showvalues = [
                (:iter, iter),
                (:loss, loss),
                (:nf, nf_per_iter[iter]),
                (:n∇ₓf, n∇ₓf_per_iter[iter]),
                (:n∇ᵤf, n∇ᵤf_per_iter[iter]),
            ],
        )

        if iter % 1 == 0
            ts = 0:0.03:T
            zs = sol.(ts)
            vs = [z.x[1][2:end] for z in zs]
            xs = [z.x[2][2:end] for z in zs]
            acts = [policy(observation(v, x, t), policy_params) for (v, x, t) in zip(vs, xs, ts)]
            goals = [goal for (goal, _) in goal_stuff.(ts)]

            Base.Filesystem.mktempdir() do dir
                electric.animate(xs, goals, acts, gravitation_position, outputdir = dir)
                # -y option overwrites existing video. See https://stackoverflow.com/questions/39788972/ffmpeg-override-output-file-if-exists
                Base.run(`ffmpeg -y -framerate 100 -i $dir/%04d.png electric_iter$iter.mp4`)
            end
        end
    end

    (
        loss_per_iter = loss_per_iter,
        policy_params_per_iter = policy_params_per_iter,
        g_per_iter = g_per_iter,
        nf_per_iter = nf_per_iter,
        n∇ₓf_per_iter = n∇ₓf_per_iter,
        n∇ᵤf_per_iter = n∇ᵤf_per_iter,
    )
end

# DiffTaichi does 200,000 iterations
stuff = run_ppg(200000)
