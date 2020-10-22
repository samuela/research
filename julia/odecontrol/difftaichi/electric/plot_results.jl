import JLSO
import ThreadPools: qmap
import Plots
import Statistics: median, quantile
import PyCall: pyimport

np = pyimport("numpy")

results_dir = "results/2020-10-19T17:57:01.569-9511d2c5f39401c2bf31d0768e475ddf3e492114-electric-many/"

@time results = qmap(1:32) do i
    JLSO.load(joinpath(results_dir, "seed$i/results.jlso"))
end
# results = [JLSO.load(joinpath(results_dir, "seed$i/results.jlso")) for i in 1:32]

# (n_iter, n_seeds)
ppg_losses_per_iter = hcat([res[:ppg_results].loss_per_iter for res in results]...)
bptt_losses_per_iter = hcat([res[:bptt_results].loss_per_iter for res in results]...)

Plots.pyplot()
Plots.PyPlotBackend()

# per iteration
every = 1000

Plots.plot(title = "DiffTaichi electric experiment")
Plots.plot!(
    1:every:size(ppg_losses_per_iter, 1),
    median(ppg_losses_per_iter[1:every:end, :], dims = 2),
    ribbon = (
        [quantile(ppg_losses_per_iter[ix, :], 0.1) for ix in 1:every:size(ppg_losses_per_iter, 1)],
        [quantile(ppg_losses_per_iter[ix, :], 0.9) for ix in 1:every:size(ppg_losses_per_iter, 1)]
    ),
    label = "PPG"
)
Plots.plot!(
    1:every:size(bptt_losses_per_iter, 1),
    median(bptt_losses_per_iter[1:every:end, :], dims = 2),
    ribbon = (
        [quantile(bptt_losses_per_iter[ix, :], 0.1) for ix in 1:every:size(bptt_losses_per_iter, 1)],
        [quantile(bptt_losses_per_iter[ix, :], 0.9) for ix in 1:every:size(bptt_losses_per_iter, 1)]
    ),
    xlabel = "Iteration",
    ylabel = "Loss",
    label = "DiffTaichi (BPTT)"
)
Plots.savefig("poop_per_iter.pdf")

# per nf
nf_eval = 1:1e6:5e7
ppg_losses_per_nf = hcat([begin
    r = res[:ppg_results]
    nfs = cumsum(r.nf_per_iter + r.n∇ₓf_per_iter + r.n∇ᵤf_per_iter)
    np.interp(nf_eval, nfs, r.loss_per_iter)
end for res in results]...)
bptt_losses_per_nf = hcat([begin
    r = res[:bptt_results]
    nfs = cumsum(r.nf_per_iter + r.n∇ₓf_per_iter + r.n∇ᵤf_per_iter)
    np.interp(nf_eval, nfs, r.loss_per_iter)
end for res in results]...)

Plots.plot(title = "DiffTaichi electric experiment")
Plots.plot!(
    nf_eval,
    median(ppg_losses_per_nf, dims = 2),
    ribbon = (
        [quantile(row, 0.1) for row in eachrow(ppg_losses_per_nf)],
        [quantile(row, 0.9) for row in eachrow(ppg_losses_per_nf)],
    ),
    label = "PPG"
)
Plots.plot!(
    nf_eval,
    median(bptt_losses_per_nf, dims = 2),
    ribbon = (
        [quantile(row, 0.1) for row in eachrow(bptt_losses_per_nf)],
        [quantile(row, 0.9) for row in eachrow(bptt_losses_per_nf)],
    ),
    xlabel = "Number of function evaluations",
    ylabel = "Loss",
    label = "DiffTaichi (BPTT)",
)
Plots.ylims!((0.0, 2.0))
Plots.savefig("poop_per_nf.pdf")
