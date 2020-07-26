include("utils.jl")

import DiffEqBase
import DiffEqSensitivity:
    solve, ODEProblem, ODEAdjointProblem, InterpolatingAdjoint, BacksolveAdjoint
import ThreadPools: qmap, tmap, bmap
import Zygote

"""Designed to be used in conjunction with loss_pullback below. Returns the
gradients wrt to `x0` and `policy_params`."""
function extract_gradients(fwd_sol, bwd_sol)
    # Logic pilfered from https://github.com/SciML/DiffEqSensitivity.jl/blob/master/src/local_sensitivity/sensitivity_interface.jl#L9.
    # We need more control than that interface gives us. Why they negate the
    # gradients is beyond me...
    p = fwd_sol.prob.p
    l = p === nothing || p === DiffEqBase.NullParameters() ? 0 : length(fwd_sol.prob.p)
    bwd_sol[end][1:length(fwd_sol.prob.u0)], -bwd_sol[end][(1:l).+length(fwd_sol.prob.u0)]
end

function extract_loss_and_xT(fwd_sol)
    fwd_sol[end][1], fwd_sol[end][2:end]
end

function count_evals(fwd_sol, bwd_sol, sensealg::InterpolatingAdjoint)
    # We do exactly as many f calls as there are function calls in the forward
    # pass, and in the backward pass we don't need to call f, but instead we
    # call ∇f.
    fwd_sol.destats.nf, bwd_sol.destats.nf
end
function count_evals(fwd_sol, bwd_sol, sensealg::BacksolveAdjoint)
    # When running the backsolve adjoint we have additional f evaluations every
    # step of the backwards pass, since we need -f to reconstruct the x path.
    fwd_sol.destats.nf + bwd_sol.destats.nf, bwd_sol.destats.nf
end

"""Returns a differentiable loss function that rolls out a policy in an
environment and calculates its cost."""
function ppg_goodies(dynamics, cost, policy, T)
    function aug_dynamics!(dz, z, policy_params, t)
        x = @view z[2:end]
        u = policy(x, policy_params)
        dz[1] = cost(x, u)
        # Note that dynamics!(dz[2:end], x, u) breaks Zygote :(
        dz[2:end] = dynamics(x, u)
    end

    # using BenchmarkTools
    # @benchmark aug_dynamics!(
    #     rand(floatT, x_dim + 1),
    #     rand(floatT, x_dim + 1),
    #     init_policy_params,
    #     0.0,
    # )

    function loss_pullback(x0, policy_params)
        z0 = vcat(0.0, x0)
        fwd_sol = solve(
            ODEProblem(aug_dynamics!, z0, (0, T), policy_params),
            Tsit5(),
            u0 = z0,
            p = policy_params,
            reltol = 1e-3,
            abstol = 1e-3,
        )

        # zT = fwd_sol[end]
        # fwd_sol, zT[1], zT[2:end], fwd_sol.destats.nf

        # TODO: this is not compatible with QuadratureAdjoint because nothing is
        # consistent... See https://github.com/SciML/DiffEqSensitivity.jl/blob/master/src/local_sensitivity/quadrature_adjoint.jl#L171.

        # This is the pullback using the augmented system and a discrete
        # gradient input at time T. Alternatively one could use the continuous
        # adjoints on the non-augmented system although this seems to be slower
        # and a less stable feature.
        function pullback(g_zT, sensealg)
            # See https://diffeq.sciml.ai/stable/analysis/sensitivity/#Syntax-1
            # and https://github.com/SciML/DiffEqSensitivity.jl/blob/master/src/local_sensitivity/sensitivity_interface.jl#L9.
            bwd_sol = solve(
                ODEAdjointProblem(
                    fwd_sol,
                    sensealg,
                    (out, x, p, t, i) -> (out[:] = g_zT),
                    [T],
                ),
                Tsit5(),
                dense = false,
                save_everystep = false,
                save_start = false,
                reltol = 1e-3,
                abstol = 1e-3,
            )

            # The first z_dim elements of bwd_sol.u are the gradient wrt z0,
            # next however many are the gradient wrt policy_params. When doing
            # the BacksolveAdjoint the final z_dim are the reconstructed z(t)
            # trajectory. We return the full bwd_sol because in some
            # circumstances we may desire information from the full solution, eg
            # where the reconstructed primals went when using BacksolveAdjoint.
            bwd_sol
        end

        # TODO:
        # * rewrite the error plots to use this version
        # * fix consumers of this api broken by the change in return type

        fwd_sol, pullback
    end

    function ez_loss_and_grad(x0, policy_params, sensealg)
        fwd_sol, vjp = loss_pullback(x0, policy_params)
        bwd_sol = vjp(vcat(1, zero(x0)), sensealg)
        loss, _ = extract_loss_and_xT(fwd_sol)
        _, g = extract_gradients(fwd_sol, bwd_sol)
        nf, n∇f = count_evals(fwd_sol, bwd_sol, sensealg)
        loss, g, (nf = nf, n∇f = n∇f)
    end

    function euler_with_cost(x0, policy_params, dt, num_steps)
        x = x0
        cost_accum = 0.0
        for _ = 1:num_steps
            u = policy(x, policy_params)
            cost_accum += dt * cost(x, u)
            x += dt * dynamics(x, u)
        end
        cost_accum
    end

    function ez_euler_bptt(x0, policy_params, dt)
        # Julia seems to do auto-rounding with floor when doing 1:num_steps. That's
        # fine for our purposes.
        num_steps = T / dt
        loss, pullback =
            Zygote.pullback((θ) -> euler_with_cost(x0, θ, dt, num_steps), policy_params)
        g, = pullback(1.0)
        loss, g, (nf = num_steps, n∇f = num_steps)
    end

    function _aggregate_batch_results(res)
        (
            mean(loss for (loss, _, _) in res),
            mean(g for (_, g, _) in res),
            (
                nf = sum(info.nf for (_, _, info) in res),
                n∇f = sum(info.n∇f for (_, _, info) in res),
            ),
        )
    end

    function ez_euler_loss_and_grad_many(x0_batch, policy_params, dt)
        _aggregate_batch_results(qmap(x0_batch) do x0
            ez_euler_bptt(x0, policy_params, dt)
        end)
    end

    function ez_loss_and_grad_many(x0_batch, policy_params, sensealg)
        # Using tmap here gives a segfault. See https://github.com/tro3/ThreadPools.jl/issues/18.
        _aggregate_batch_results(qmap(x0_batch) do x0
            ez_loss_and_grad(x0, policy_params, sensealg)
        end)
    end

    (
        aug_dynamics! = aug_dynamics!,
        loss_pullback = loss_pullback,
        ez_loss_and_grad = ez_loss_and_grad,
        ez_loss_and_grad_many = ez_loss_and_grad_many,
        ez_euler_bptt = ez_euler_bptt,
        ez_euler_loss_and_grad_many = ez_euler_loss_and_grad_many,
    )
end
