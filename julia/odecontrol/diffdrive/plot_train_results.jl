import JLSO
import Plots
import ProgressMeter: @showprogress
import Random

Random.seed!(123)
ENV["GKSwstype"] = "nul"

results = JLSO.load("diffdrive_train_results.jlso")
ppg_params = results[:interp_results].policy_params_per_iter
euler_params = results[:euler_results].policy_params_per_iter
@assert size(ppg_params) == size(euler_params)

p = Plots.plot(interp_results.loss_per_iter, label = "loss")
Plots.savefig(p, "diffdrive_loss_per_iter.pdf")

p = Plots.plot(
    map(mean, eachrow(abs.(interp_results.policy_params_per_iter))),
    label = "mean abs of each weight",
)
Plots.savefig(p, "diffdrive_weights_per_iter.pdf")

p = Plots.plot(map(norm, eachrow(interp_results.g_per_iter)), label = "norm of gradients")
Plots.savefig(p, "diffdrive_grads_per_iter.pdf")

p = Plots.plot(
    map(mean, eachrow(abs.(interp_results.g_per_iter))),
    label = "mean abs of gradients",
    yaxis = :log10,
)
Plots.savefig(p, "diffdrive_grads_per_iter.pdf")

# NOTE: Make sure this is the same as in train.jl!
dynamics, cost, sample_x0, obs = DiffDriveEnv.diffdrive_env(floatT, 1.0f0, 0.5f0)
learned_policy_goodies = ppg_goodies(dynamics, cost, policy, T)
x0_test_batch = [sample_x0() for _ = 1:10]
# num_hidden = 32
# policy = FastChain(
#     (x, _) -> obs(x),
#     FastDense(7, num_hidden, tanh),
#     FastDense(num_hidden, num_hidden, tanh),
#     FastDense(num_hidden, 2),
# )

function rollout(x0, θ)
    sol = solve(
        ODEProblem((x, p, t) -> dynamics(x, policy(x, θ)), x0, (0, T)),
        Tsit5(),
        saveat = 0.1,
    )
    xs = [s[1] for s in sol.u]
    ys = [s[2] for s in sol.u]
    (xs, ys)
end

@time ppg_trajs = qmap(collect(eachrow(ppg_params))) do θ
    [rollout(x0, θ) for x0 in x0_test_batch]
end
@time euler_trajs = qmap(collect(eachrow(euler_params))) do θ
    [rollout(x0, θ) for x0 in x0_test_batch]
end

anim = Plots.Animation()
@showprogress for iter = 1:size(ppg_params, 1)
    p = Plots.plot(
        title = "Iteration $iter",
        legend = true,
        aspect_ratio = :equal,
        xlims = (-7.5, 7.5),
        ylims = (-7.5, 7.5),
    )
    for (xs, ys) in euler_trajs[iter]
        Plots.plot!(xs, ys, color = :blue, linestyle = :dash, label = nothing)
    end
    for (xs, ys) in ppg_trajs[iter]
        Plots.plot!(xs, ys, color = :red, label = nothing)
    end
    Plots.scatter!(
        [x0[1] for x0 in x0_test_batch],
        [x0[2] for x0 in x0_test_batch],
        color = :grey,
        markersize = 5,
        label = "Initial conditions",
    )
    Plots.plot!([], color = :blue, linestyle = :dash, label = "Euler BPTT")
    Plots.plot!([], color = :red, label = "PPG (ours)")
    Plots.frame(anim)
end
Plots.mp4(anim, "diffdrive.mp4")
