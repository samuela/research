import DiffEqBase
import DiffEqSensitivity: solve, ODEProblem, ODEAdjointProblem

"""Designed to be used in conjunction with loss_pullback below. Returns the """
function extract_gradients(fwd_sol, bwd_sol)
    # Logic pilfered from https://github.com/SciML/DiffEqSensitivity.jl/blob/master/src/local_sensitivity/sensitivity_interface.jl#L9.
    # We need more control than that interface gives us. Why they negate the
    # gradients is beyond me...
    p = fwd_sol.prob.p
    l = p === nothing || p === DiffEqBase.NullParameters() ? 0 : length(fwd_sol.prob.p)
    -bwd_sol[end][1:length(fwd_sol.prob.u0)], -bwd_sol[end][(1:l) .+ length(fwd_sol.prob.u0)]
end

function extract_loss_and_xT(fwd_sol)
    fwd_sol[end][1], fwd_sol[end][2:end]
end

"""Returns a differentiable loss function that rolls out a policy in an
environment and calculates its cost."""
function ppg_goodies(dynamics, cost, policy)
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

    function loss_pullback(x0, policy_params, sensealg)
        z0 = vcat(0.0, x0)
        fwd_sol = solve(
            ODEProblem(aug_dynamics!, z0, (0, T), policy_params),
            Tsit5(),
            u0 = z0,
            p = policy_params,
            # TODO: is this necessary?
            sensealg = sensealg,
        )

        # zT = fwd_sol[end]
        # fwd_sol, zT[1], zT[2:end], fwd_sol.destats.nf

        # This is the pullback using the augmented system and a discrete
        # gradient input at time T. Alternatively one could use the continuous
        # adjoints on the non-augmented system although this seems to be slower
        # and a less stable feature.
        function pullback(g_zT)
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
        # * write an "easy version" and then use that in diffdrive/train.jl
        # * rewrite the error plots to use this version
        # * fix consumers of this api broken by the change in return type

        fwd_sol, pullback
    end

    (aug_dynamics! = aug_dynamics!, loss_pullback = loss_pullback)
end
