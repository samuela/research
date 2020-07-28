"""An differential drive robot environment.

See http://planning.cs.uiuc.edu/node659.html.
"""

#=
struct DDrive{T<:AbstractFloat} #where T #<:AbstractFloat
    wheelbase::T
    wheel_radius::T
end

function dynamics(dd::DDrive{T}, state, u) where T
    x, y, θ, ω_l, ω_r = state
    sinθ, cosθ = sincos(θ)
    [
     dd.wheel_radius / 2 * (ω_l + ω_r) * cosθ,
     dd.wheel_radius / 2 * (ω_l + ω_r) * sinθ,
     dd.wheel_radius / dd.wheelbase * (ω_r - ω_l),
     u[1],
     u[2],
    ]
end

function cost(state, u)
    x, y, θ, ω_l, ω_r = state
    x^2 + y^2 + 0.1 * (ω_l^2 + ω_r^2 + u[1]^2 + u[2]^2)
end

function sample_x0(dd::DDrive{T}) where T
    [
     rand(T) * 10 - 5,
     rand(T) * 10 - 5,
     rand(T) * twopi,
     # randn(floatT),
     # randn(floatT),
     zero(T),
     zero(T),
    ]::Array{floatT}
end

function observation(state)
    x, y, θ, ω_l, ω_r = state
    sinθ, cosθ = sincos(θ)
    [x, y, θ % twopi, ω_l, ω_r, sinθ, cosθ]
end
=#

module DiffDriveEnv

function diffdrive_env(floatT, wheelbase, wheel_radius)
    twopi = convert(floatT, 2π)

    function dynamics(state, u)
        x, y, θ, ω_l, ω_r = state
        ω_l = 0.9 * ω_l
        ω_r = 0.9 * ω_r
        sinθ, cosθ = sincos(θ)
        [
         wheel_radius / 2 * (ω_l + ω_r) * cosθ,
         wheel_radius / 2 * (ω_l + ω_r) * sinθ,
         wheel_radius / wheelbase * (ω_r - ω_l),
         u[1],
         u[2],
        ]
    end

    function cost(state, u)
        x, y, θ, ω_l, ω_r = state
        x^2 + y^2 + 0.1 * (ω_l^2 + ω_r^2 + u[1]^2 + u[2]^2)
    end

    function sample_x0()
        [
         rand(floatT) * 10 - 5,
         rand(floatT) * 10 - 5,
         rand(floatT) * twopi,
         # randn(floatT),
         # randn(floatT),
         0,
         0,
        ]::Array{floatT}
    end

    function observation(state)
        x, y, θ, ω_l, ω_r = state
        sinθ, cosθ = sincos(θ)
        [x, y, θ % twopi, ω_l, ω_r, sinθ, cosθ]
    end

    dynamics, cost, sample_x0, observation
end

end
