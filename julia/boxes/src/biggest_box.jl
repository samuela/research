import Base: +, -, *, ==, <=, <, big, convert, isfinite, isless, iszero, iterate, length, show, zero
import Ipopt
import JuMP
import MathOptInterface: MAX_SENSE
import NLopt
import Printf: @printf

include("boxes.jl")
include("intervals.jl")

abstract type TracedExpr end

"""
These are constants that we meet along the way.

No `world` pointer here so that we can define `Base.zero` on `TracedExpr`s. This
isn't strictly a requirement anymore now that we stopped using
IntervalArithmetic.jl and rolled our own.
"""
struct TracedConstant <: TracedExpr
  value
end

"""These are the "leaves" of the computation graph and have names."""
struct TracedLeaf <: TracedExpr
  world
  name
  value
end

struct TracedMul <: TracedExpr
  world
  lhs::TracedExpr
  rhs::TracedExpr
  value

  TracedMul(lhs::TracedExpr, rhs::TracedExpr) = new(same_world(lhs, rhs), lhs, rhs, lhs.value * rhs.value)
end

struct TracedAdd <: TracedExpr
  world
  lhs::TracedExpr
  rhs::TracedExpr
  value

  TracedAdd(lhs::TracedExpr, rhs::TracedExpr) = new(same_world(lhs, rhs), lhs, rhs, lhs.value + rhs.value)
end

struct TracedSub <: TracedExpr
  world
  lhs::TracedExpr
  rhs::TracedExpr
  value

  TracedSub(lhs::TracedExpr, rhs::TracedExpr) = new(same_world(lhs, rhs), lhs, rhs, lhs.value - rhs.value)
end

"Assert that two TracedExpr's have the same world, and then return that world."
same_world(::TracedConstant, y::TracedExpr) = y.world
same_world(x::TracedExpr, ::TracedConstant) = x.world
same_world(x::TracedExpr, y::TracedExpr) = (@assert x.world === y.world; x.world)
# NOTE: do we need `same_world(::TracedConstant, ::TracedConstant)`?

# Note: this assumes that the value of TracedThings is Float64.
# zero(::Type{T}) where {T <: TracedExpr} = TracedConstant(0.0)

big(x::TracedExpr) = big(x.value)
iszero(x::TracedExpr) = iszero(x.value)
isfinite(x::TracedExpr) = isfinite(x.value)

# Necessary for broadcasting like `in_box.(y_boxes, Ref(TC(y)))`.
# length(x::TracedExpr) = length(x.value)
# iterate(x::TracedExpr) = iterate(x.value)
# iterate(x::TracedExpr, state) = iterate(x.value, state)

+(lhs::TracedExpr, rhs::TracedExpr) = TracedAdd(lhs, rhs)
-(lhs::TracedExpr, rhs::TracedExpr) = TracedSub(lhs, rhs)
*(lhs::TracedExpr, rhs::TracedExpr) = TracedMul(lhs, rhs)

# TODO actually respect RoundingMode
# +(lhs::TracedExpr, rhs::TracedExpr, ::RoundingMode) = TracedAdd(lhs, rhs)
# *(lhs::TracedExpr, rhs::TracedExpr, ::RoundingMode) = TracedMul(lhs, rhs)

abstract type Constraint end
# TODO constructors that check worlds are ===
struct LessThan <: Constraint
  lhs
  rhs
end
struct GreaterThan <: Constraint
  lhs
  rhs
end
struct LessThanOrEq <: Constraint
  lhs
  rhs
end
struct GreaterThanOrEq <: Constraint
  lhs
  rhs
end
struct Equal <: Constraint
  lhs
  rhs
end
struct NotEqual <: Constraint
  lhs
  rhs
end
function <(lhs::Float64, rhs::TracedExpr)
  res = lhs < rhs.value
  push!(rhs.world, (res ? LessThan : GreaterThanOrEq)(TracedConstant(lhs), rhs))
  res
end
function <(lhs::TracedExpr, rhs::Float64)
  res = lhs.value < rhs
  push!(lhs.world, (res ? LessThan : GreaterThanOrEq)(lhs, TracedConstant(rhs)))
  res
end
function <(lhs::TracedExpr, rhs::TracedExpr)
  res = lhs.value < rhs.value
  push!(lhs.world, (res ? LessThan : GreaterThanOrEq)(lhs, rhs))
  res
end
function isless(lhs::TracedExpr, rhs::TracedExpr)
  res = lhs.value < rhs.value
  push!(lhs.world, (res ? LessThan : GreaterThanOrEq)(lhs, rhs))
  res
end
function ==(lhs::Float64, rhs::TracedExpr)
  res = lhs == rhs.value
  push!(rhs.world, (res ? Equal : NotEqual)(TracedConstant(lhs), rhs))
  res
end
function ==(lhs::TracedExpr, rhs::Float64)
  res = lhs.value == rhs
  push!(lhs.world, (res ? Equal : NotEqual)(lhs, TracedConstant(rhs)))
  res
end
function <=(lhs::Float64, rhs::TracedExpr)
  res = lhs <= rhs.value
  push!(rhs.world, (res ? LessThanOrEq : GreaterThan)(TracedConstant(lhs), rhs))
  res
end
function <=(lhs::TracedExpr, rhs::Float64)
  res = lhs.value <= rhs
  push!(lhs.world, (res ? LessThanOrEq : GreaterThan)(lhs, TracedConstant(rhs)))
  res
end
function <=(lhs::TracedExpr, rhs::TracedExpr)
  res = lhs.value <= rhs.value
  push!(same_world(lhs, rhs), (res ? LessThanOrEq : GreaterThan)(lhs, rhs))
  res
end

# Make sure that these add constraints to the world!
lt_zero(x::TracedExpr) = x < TracedConstant(zero(x.value))
gt_zero(x::TracedExpr) = x > TracedConstant(zero(x.value))
lteq_zero(x::TracedExpr) = x <= TracedConstant(zero(x.value))
gteq_zero(x::TracedExpr) = x >= TracedConstant(zero(x.value))

show(io::IO, x::TracedConstant) = show(io, x.value)
show(io::IO, x::TracedLeaf) = print(io, x.name)
# show(io::IO, x::TracedLeaf) = @printf io "%s(=%s)" x.name repr(x.value)
show(io::IO, x::TracedMul) = @printf io "%s * %s" repr(x.lhs) repr(x.rhs)
show(io::IO, x::TracedAdd) = @printf io "%s + %s" repr(x.lhs) repr(x.rhs)
show(io::IO, x::TracedSub) = @printf io "%s - %s" repr(x.lhs) repr(x.rhs)
show(io::IO, x::LessThan) = @printf io "%s < %s" repr(x.lhs) repr(x.rhs)
show(io::IO, x::GreaterThan) = @printf io "%s > %s" repr(x.lhs) repr(x.rhs)
show(io::IO, x::LessThanOrEq) = @printf io "%s <= %s" repr(x.lhs) repr(x.rhs)
show(io::IO, x::GreaterThanOrEq) = @printf io "%s >= %s" repr(x.lhs) repr(x.rhs)
show(io::IO, x::Equal) = @printf io "%s == %s" repr(x.lhs) repr(x.rhs)
show(io::IO, x::NotEqual) = @printf io "%s =/= %s" repr(x.lhs) repr(x.rhs)

"""Consume a `TracedExpr` and evaluate it to an `Interval` value, by injecting
`Interval(TracedLeaf(x_lo), TracedLeaf(x_hi))` in place of `TracedLeaf(x)`. Also
replace the world with `w`. In other words, any comparisons done in the process
of computing these intervals will add constraints to `w`."""
interval_eval(_, expr::TracedConstant) = Interval(expr, expr)
interval_eval(w, expr::TracedLeaf) = Interval(TracedLeaf(w, expr.name * "_lo", expr.value),
                                              TracedLeaf(w, expr.name * "_hi", expr.value))
interval_eval(w, expr::TracedMul) = interval_eval(w, expr.lhs) * interval_eval(w, expr.rhs)
interval_eval(w, expr::TracedAdd) = interval_eval(w, expr.lhs) + interval_eval(w, expr.rhs)
interval_eval(w, expr::TracedSub) = interval_eval(w, expr.lhs) - interval_eval(w, expr.rhs)

# TODO this is a hack so that we can create pairs with intervals
# See
# * https://discourse.julialang.org/t/x-y-works-but-x-y-doesnt/62703
# * https://github.com/JuliaIntervals/IntervalArithmetic.jl/issues/476
# Base.convert(::Type{Interval{T}}, x::Interval{T}) where {T <: TracedExpr} = x
# Base.convert(::Type{Interval{TracedExpr}}, x::Interval{<:TracedExpr}) = Interval(convert(TracedExpr, x.lo)::TracedExpr, convert(TracedExpr, x.hi)::TracedExpr)::Interval{TracedExpr}

"""Normalize constraints into g(x) <= 0 form. Drop the < vs <= distinction. This
should fail with a method-not-found error if we ever accidentally attempt to
normalize an equality constraint."""
normalize_constraint(c::LessThan) = c.lhs - c.rhs
normalize_constraint(c::GreaterThan) = c.rhs - c.lhs
normalize_constraint(c::LessThanOrEq) = c.lhs - c.rhs
normalize_constraint(c::GreaterThanOrEq) = c.rhs - c.lhs

quotify(_, expr::TracedConstant) = expr.value
quotify(env, expr::TracedLeaf) = env[expr.name]
quotify(env, expr::TracedMul) = :($(quotify(env, expr.lhs)) * $(quotify(env, expr.rhs)))
quotify(env, expr::TracedAdd) = :($(quotify(env, expr.lhs)) + $(quotify(env, expr.rhs)))
quotify(env, expr::TracedSub) = :($(quotify(env, expr.lhs)) - $(quotify(env, expr.rhs)))

eval(_, expr::TracedConstant) = expr.value
eval(env, expr::TracedLeaf) = env[expr.name]
eval(env, expr::TracedMul) = eval(env, expr.lhs) * eval(env, expr.rhs)
eval(env, expr::TracedAdd) = eval(env, expr.lhs) + eval(env, expr.rhs)
eval(env, expr::TracedSub) = eval(env, expr.lhs) - eval(env, expr.rhs)

function biggest_box_2d(f, x, y, bounds)
  # Run `f` and collect constraints into `w1`.
  w1 = Constraint[]
  f_eval = f(TracedLeaf(w1, "x", x), TracedLeaf(w1, "y", y))
  # Evaluate the constraints in `w1` with interval arithmetic and collect new
  # constraints into `w2`.
  w2 = Constraint[]
  exprs1 = map(c -> interval_eval(w2, normalize_constraint(c)).hi, w1)
  exprs2 = map(c -> normalize_constraint(c), w2)

  model = JuMP.Model(NLopt.Optimizer)
  JuMP.set_optimizer_attribute(model, "algorithm", :LD_SLSQP)

  # Force the optimizer to start at an initial point that has non-zero gradient.
  # TODO: better to do bisection for the initial point?
  # ϵ1 is the initial half-width of the box. ϵ2 is a nudge factor to ensure that
  # we stay within the variable bounds in the worst case.
  ϵ1 = 0.001
  ϵ2 = 1e-6
  # min <= lo <= x <= hi <= max
  x_lo = JuMP.@variable(model, base_name="x_lo", lower_bound=bounds.mins[1], upper_bound=x, start=max(x - ϵ1, bounds.mins[1] + ϵ2))
  x_hi = JuMP.@variable(model, base_name="x_hi", lower_bound=x, upper_bound=bounds.maxs[1], start=min(x + ϵ1, bounds.maxs[1] - ϵ2))
  y_lo = JuMP.@variable(model, base_name="y_lo", lower_bound=bounds.mins[2], upper_bound=y, start=max(y - ϵ1, bounds.mins[2] + ϵ2))
  y_hi = JuMP.@variable(model, base_name="y_hi", lower_bound=y, upper_bound=bounds.maxs[2], start=min(y + ϵ1, bounds.maxs[2] - ϵ2))
  for expr in [exprs1; exprs2]
    q = quotify(Dict("x_lo" => x_lo, "x_hi" => x_hi, "y_lo" => y_lo, "y_hi" => y_hi), expr)
    JuMP.add_NL_constraint(model, :($(q) <= 0))
  end

  # Note: this is more generic than necessary at the moment.
  widths = [:($(x_hi) - $(x_lo)), :($(y_hi) - $(y_lo))]
  JuMP.set_NL_objective(model, MAX_SENSE, Expr(:call, :*, widths...))
  JuMP.optimize!(model)

  f_eval, Box([JuMP.value(x_lo), JuMP.value(y_lo)], [JuMP.value(x_hi), JuMP.value(y_hi)])
end
