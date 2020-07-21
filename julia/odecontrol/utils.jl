"""Returns a differentiable loss function that rolls out a policy in an
environment and calculates its cost."""
function policy_loss(dynamics, cost, policy, sensealg)
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

    """Loss function that rolls out the augmented dynamics, and returns the mean
    cost.

    Here `data` should be an iterable initial conditions."""
    function loss_f(policy_params, data)
        # TODO: use the ensemble thing
        mean([
            begin
                z0 = vcat(0.0, x0)
                rollout = solve(
                    ODEProblem(aug_dynamics!, z0, (0, T), policy_params),
                    Tsit5(),
                    u0 = z0,
                    p = policy_params,
                    sensealg = sensealg,
                )
                Array(rollout)[1, end]
            end for x0 in data
        ])
    end

    loss_f
end
