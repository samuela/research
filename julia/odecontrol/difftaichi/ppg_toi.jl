import DiffEqBase
import DiffEqBase: DiscreteCallback, DynamicalODEProblem
import DiffEqSensitivity:
    solve,
    ODEProblem,
    ODEAdjointProblem,
    InterpolatingAdjoint,
    AdjointSensitivityIntegrand
import Zygote
import RecursiveArrayTools: ArrayPartition

struct TOIStuff
    conditions
    affect
    time_epsilon::Float64
end

struct TOISolution
    # An array of ODESolution's. Sometimes we get `OrdinaryDiffEq.ODECompositeSolution` instead of
    # `DiffEqBase.ODESolution` and they don't have a subtype relationship, so we resort to just `Array`.
    solutions::Array
end

function (sol::TOISolution)(t)
    @assert sol.solutions[1].t[1] <= t <= sol.solutions[end].t[end]
    # TODO: what does this do in the cases where t < 0 and t > T?
    ix = searchsortedlast([s.t[1] for s in sol.solutions], t)
    subsol = sol.solutions[ix]
    # There is some small 2 * time_epsilon sized interval after the solution stops, that t may fall into. In that case
    # we just return the last value in the solution.
    if subsol.t[end] < t
        subsol.u[end]
    else
        subsol(t)
    end
end

function ppg_toi_goodies(v_dynamics, x_dynamics, cost, policy, toi, T)
    # x and v are ArrayPartitions(cost::Real, 2-d Array).
    # Our trick for shoehorning cost accumulation into a DynamicalODEProblem is to include it in both x and v. Then we
    # update it in the aug_dyn_x dynamics. The space for it in v is unused and left at zero.
    aug_dyn_v(v_ap, x_ap, policy_params, t) = begin
        # TODO this might be the problem. Need to isolate minimal reproduction.
        v = v_ap.x[2]
        x = x_ap.x[2]
        # Don't know the original shapes of x, v which makes this a bit difficult.
        # v = reshape(@view(v_ap[2:end]), (:, 2))
        # x = reshape(@view(x_ap[2:end]), (:, 2))
        u = policy(x, v, policy_params, t)
        ArrayPartition(zeros(), v_dynamics(v, x, u))
    end
    aug_dyn_x(v_ap, x_ap, policy_params, t) = begin
        v = v_ap.x[2]
        x = x_ap.x[2]
        # v = reshape(@view(v_ap[2:end]), (:, 2))
        # x = reshape(@view(x_ap[2:end]), (:, 2))
        u = policy(x, v, policy_params, t)
        # See https://github.com/FluxML/Zygote.jl/issues/800.
        ArrayPartition(fill(cost(v, x, u)), x_dynamics(v, x, u))
    end

    callback = DiscreteCallback((_, _, _) -> true, (integrator) -> begin
        # In some cases (eg toi_test6.jl) there is a zero-crossing, but the integrators are sufficiently smart as to
        # skip it entirely. Ie, you cross zero twice and so you would never notice without some more careful testing.
        test_times = range(integrator.tprev, integrator.t, length=10)
        # This is an Array of ArrayPartition(ArrayPartition(cost, v), ArrayPartition(cost, x)).
        test_vals = integrator.(test_times)

        # There's a little tiny droplet of optimization that could be squeezed out here. We could be lazier.
        # For each condition, find the index of the time span with the first down-crossing. `nothing` if no
        # down-crossing is found.
        ixs = filter((a) -> !isnothing(a[1]), [begin
            cs = [condition(vx_aug.x[1].x[2], vx_aug.x[2].x[2]) for vx_aug in test_vals]
            # Only fire for down-crossings: positive -> negative.
            (findfirst((j) -> cs[j] > 0 > cs[j + 1], 1:(length(cs) - 1)), i)
        end for (i, condition) in enumerate(toi.conditions)])

        if length(ixs) > 0
            (t_ix, condition_ix) = ixs[argmin(map(first, ixs))]

            # See https://github.com/SciML/DiffEqBase.jl/blob/d4973e21ff31dc1d355e84ae2b4c1d3c9546b6b2/src/callbacks.jl#L673.
            event_t = DiffEqBase.bisection(
                (t) -> begin
                    vx_aug = integrator(t)
                    toi.conditions[condition_ix](vx_aug.x[1].x[2], vx_aug.x[2].x[2])
                end,
                (test_times[t_ix], test_times[t_ix + 1]),
                isone(integrator.tdir)
            )

            # If, on the off chance, event_t - time_epsilon < tprev, this will return an error. Soluton is
            # to set time_epsilon smaller. Taking the max is technically messing up time a little bit, but
            # saves us a bunch of trouble.
            DiffEqBase.change_t_via_interpolation!(integrator, max(event_t - toi.time_epsilon, integrator.tprev))
            DiffEqBase.terminate!(integrator)
        end
    end)

    # See https://discourse.julialang.org/t/why-the-separation-of-odeproblem-and-solve-in-differentialequations-jl/43737
    # for a discussion of the performance of the pullbacks.
    function loss_pullback(v0, x0, policy_params, solvealg, solve_kwargs)
        # So we need to do some weird hacks to get correct gradients when
        # dealing with events. DiffEqSensitivity does not return the correct
        # gradients when using ContinuousCallbacks. Instead we manually detect
        # our callbacks and use the TOI trick around them. However this is
        # non-trivial and requires us to do our own little AD to get everything
        # working properly.
        done = false
        current_t = 0.0
        current_vx = ArrayPartition(
            ArrayPartition(zeros(), v0),
            ArrayPartition(zeros(), x0))
        tape = []
        while !done
            # There's some very small chance that we TOI jump past the final time T. I don't want to find out what
            # DifferentialEquations.jl does in that case.
            @assert current_t < T

            fwd_sol = solve(
                DynamicalODEProblem(
                    aug_dyn_v,
                    aug_dyn_x,
                    current_vx.x[1],
                    current_vx.x[2],
                    (current_t, T),
                    policy_params
                ),
                solvealg,
                p = policy_params;
                callback = callback,
                solve_kwargs...,
            )

            if fwd_sol.retcode == :Terminated
                current_t = fwd_sol.t[end] + 2 * toi.time_epsilon
                current_vx[:], toi_pullback = Zygote.pullback(
                    (vx) -> begin
                        v = vx.x[1].x[2]
                        x = vx.x[2].x[2]
                        accum_cost = vx.x[2].x[1]
                        new_v, new_x = toi.affect(v, x, 2 * toi.time_epsilon)
                        ArrayPartition(
                            ArrayPartition(zeros(), new_v),
                            ArrayPartition(accum_cost, new_x))
                    end,
                    fwd_sol[end])
                push!(tape, (fwd_sol, toi_pullback))
            else
                @assert fwd_sol.retcode == :Success
                push!(tape, (fwd_sol, (x) -> (x, )))
                done = true
            end
        end

        # This is the pullback using the augmented system and a discrete
        # gradient input at time T. Alternatively one could use the continuous
        # adjoints on the non-augmented system although this seems to be slower
        # and a less stable feature.
        function pullback(g_zT, sensealg::InterpolatingAdjoint)
            # See https://diffeq.sciml.ai/stable/analysis/sensitivity/#Syntax-1
            # and https://github.com/SciML/DiffEqSensitivity.jl/blob/master/src/local_sensitivity/sensitivity_interface.jl#L9.
            g_z = g_zT
            g_p = zero(tape[1][1].prob.p)
            n∇ₓf = 0
            n∇ᵤf = 0
            for (fwd_sol, pb) in reverse(tape)
                # Backprop through the TOI step...
                (g_z, ) = pb(g_z)
                # Backprop through the ODE solve...
                bwd_sol = solve(
                    ODEAdjointProblem(
                        fwd_sol,
                        sensealg,
                        (out, x, p, t, i) -> (out[:] = g_z),
                        [fwd_sol.t[end]],
                    ),
                    solvealg;
                    dense = false,
                    save_everystep = false,
                    save_start = false,
                )
                @error "yay bwd_sol"
                # We do exactly as many f calls as there are function calls in the
                # forward pass, and in the backward pass we don't need to call f,
                # but instead we call ∇f.
                n∇ₓf += bwd_sol.destats.nf
                n∇ᵤf += bwd_sol.destats.nf

                # The first z_dim elements of bwd_sol.u are the gradient wrt z0,
                # next however many are the gradient wrt policy_params.
                g_z = bwd_sol[end][1:length(fwd_sol.prob.u0)]

                # No clue why DiffEqSensitivity negates this...
                g_p -= bwd_sol[end][(1:length(g_p)).+length(fwd_sol.prob.u0)]
            end

            (
                g_z0 = g_z,
                g_p = g_p,
                nf = 0,
                n∇ₓf = n∇ₓf,
                n∇ᵤf = n∇ᵤf,
            )
        end

        TOISolution(map(first, tape)), pullback
    end

    loss_pullback
end
