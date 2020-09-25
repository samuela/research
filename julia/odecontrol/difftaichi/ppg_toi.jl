# include("../ppg.jl")

import DiffEqBase
import DiffEqBase: DiscreteCallback
import DiffEqSensitivity:
    solve,
    ODEProblem,
    ODEAdjointProblem,
    InterpolatingAdjoint,
    AdjointSensitivityIntegrand
import Zygote

struct TOIStuff
    condition
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

function ppg_toi_goodies(dynamics, cost, policy, toi, T)
    function aug_dynamics(z, policy_params, t)
        x = @view z[2:end]
        u = policy(x, t, policy_params)
        [cost(x, u); dynamics(x, u)]
    end

    # See https://discourse.julialang.org/t/why-the-separation-of-odeproblem-and-solve-in-differentialequations-jl/43737
    # for a discussion of the performance of the pullbacks.
    function loss_pullback(x0, policy_params, solvealg, solve_kwargs)
        # So we need to do some weird hacks to get correct gradients when
        # dealing with events. DiffEqSensitivity does not return the correct
        # gradients when using ContinuousCallbacks. Instead we manually detect
        # our callbacks and use the TOI trick around them. However this is
        # non-trivial and requires us to do our own little AD to get everything
        # working properly.
        done = false
        current_t = 0.0
        current_z = vcat(0.0, x0)
        tape = []
        while !done
            # There's some very small chance that we TOI jump past the final time T. I don't want to find out what
            # DifferentialEquations.jl does in that case.
            @assert current_t < T
            fwd_sol = solve(
                ODEProblem(aug_dynamics, current_z, (current_t, T), policy_params),
                solvealg,
                u0 = current_z,
                p = policy_params;
                callback = DiscreteCallback((u, t, integrator) -> toi.condition(u[2:end]) < 0, (integrator) -> begin
                    # Only fire for down-crossings: positive -> negative.
                    if toi.condition(integrator.uprev[2:end]) > 0
                        # See https://github.com/SciML/DiffEqBase.jl/blob/d4973e21ff31dc1d355e84ae2b4c1d3c9546b6b2/src/callbacks.jl#L673.
                        event_t = DiffEqBase.bisection(
                            (t) -> toi.condition(integrator(t)[2:end]),
                            (integrator.tprev, integrator.t), isone(integrator.tdir)
                        )

                        # If, on the off chance, event_t - time_epsilon < tprev, this will return an error. Soluton is
                        # to set time_epsilon smaller. Taking the max is technically messing up time a little bit, but
                        # saves us a bunch of trouble.
                        DiffEqBase.change_t_via_interpolation!(integrator, max(event_t - toi.time_epsilon, integrator.tprev))
                        DiffEqBase.terminate!(integrator)
                    end
                end),
                solve_kwargs...,
            )

            if fwd_sol.retcode == :Terminated
                current_t = fwd_sol.t[end] + 2 * toi.time_epsilon
                current_z[:], toi_pullback = Zygote.pullback(
                    (z) -> [z[1]; toi.affect(z[2:end], 2 * toi.time_epsilon)],
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

    (
        aug_dynamics = aug_dynamics,
        loss_pullback = loss_pullback,
    )
end
