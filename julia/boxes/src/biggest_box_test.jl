import Base: +, -, *, ==, <=, <, big, convert, isfinite, isless, iszero, show, zero
import InteractiveUtils: @code_typed
import Ipopt
import JuMP
import MathOptInterface: MAX_SENSE
import NLopt
import Printf: @printf
import Test: @test

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

is_free1(x, y) = (-2.0 <= x) && (x <= 2.0) && (-2.0 <= y) && (y <= 2.0) && (x*x + y*y >= 1.0)
is_free2(x, y) = (-2.0 <= x <= 2.0) && (-2.0 <= y <= 2.0) && (x*x + y*y >= 1.0)
function is_free3(x, y)
  if !(-2.0 <= x <= 2.0)
    return false
  end
  if !(-2.0 <= y <= 2.0)
    return false
  end
  x*x + y*y >= 1.0
end
is_free4(x, y) = x*x + y*y >= 1.0

let w = Constraint[]
  @test is_free1(TracedLeaf(w, "x", -1.5), TracedLeaf(w, "y", 1.5))
  @test repr(w) == "Constraint[-2.0 <= x, x <= 2.0, -2.0 <= y, y <= 2.0, 1.0 <= x * x + y * y]"
end
let w = Constraint[]
  @test is_free2(TracedLeaf(w, "x", -1.5), TracedLeaf(w, "y", 1.5))
  @test repr(w) == "Constraint[-2.0 <= x, x <= 2.0, -2.0 <= y, y <= 2.0, 1.0 <= x * x + y * y]"
end
let w = Constraint[]
  @test is_free3(TracedLeaf(w, "x", 2.0), TracedLeaf(w, "y", 1.5))
  @test repr(w) == "Constraint[-2.0 <= x, x <= 2.0, -2.0 <= y, y <= 2.0, 1.0 <= x * x + y * y]"
end
# Test exactly on the left boundary of x...
let w = Constraint[]
  @test is_free3(TracedLeaf(w, "x", -2.0), TracedLeaf(w, "y", 1.5))
  @test repr(w) == "Constraint[-2.0 <= x, x <= 2.0, -2.0 <= y, y <= 2.0, 1.0 <= x * x + y * y]"
end
# Test exactly on the right boundary of x...
let w = Constraint[]
  @test is_free3(TracedLeaf(w, "x", 2.0), TracedLeaf(w, "y", 1.5))
  @test repr(w) == "Constraint[-2.0 <= x, x <= 2.0, -2.0 <= y, y <= 2.0, 1.0 <= x * x + y * y]"
end
# Test exactly on the left boundary of y...
let w = Constraint[]
  @test is_free3(TracedLeaf(w, "x", 1.5), TracedLeaf(w, "y", -2))
  @test repr(w) == "Constraint[-2.0 <= x, x <= 2.0, -2.0 <= y, y <= 2.0, 1.0 <= x * x + y * y]"
end
# Test both on the boundary
let w = Constraint[]
  @test is_free3(TracedLeaf(w, "x", 2), TracedLeaf(w, "y", -2))
  @test repr(w) == "Constraint[-2.0 <= x, x <= 2.0, -2.0 <= y, y <= 2.0, 1.0 <= x * x + y * y]"
end
# Test inside the circle
let w = Constraint[]
  @test !is_free1(TracedLeaf(w, "x", 0.1), TracedLeaf(w, "y", 0.0))
  @test repr(w) == "Constraint[-2.0 <= x, x <= 2.0, -2.0 <= y, y <= 2.0, 1.0 > x * x + y * y]"
end
# Test outside the box
let w = Constraint[]
  @test !is_free1(TracedLeaf(w, "x", 2.1), TracedLeaf(w, "y", 0.0))
  @test repr(w) == "Constraint[-2.0 <= x, x > 2.0]"
end
let w = Constraint[]
  @test !is_free1(TracedLeaf(w, "x", -2.1), TracedLeaf(w, "y", 0.0))
  @test repr(w) == "Constraint[-2.0 > x]"
end

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

# @test promote_type(Interval{TracedConstant}, Interval{TracedLeaf}) == Interval{TracedExpr}

# let w = Constraint[]
#   @test is_free4(TracedLeaf(w, "x", 1.75), TracedLeaf(w, "y", 1.5))
#   norm_constraints = map(normalize_constraint, w)
#   @test repr(map(expr -> interval_eval(expr).hi, norm_constraints)) == "TracedSub[1.0 - x_lo * x_lo + y_lo * y_lo]"
# end
# let w = Constraint[]
#   @test is_free4(TracedLeaf(w, "x", -1.5), TracedLeaf(w, "y", 1.5))
#   norm_constraints = map(normalize_constraint, w)
#   @test repr(map(expr -> interval_eval(expr).hi, norm_constraints)) == "TracedSub[1.0 - x_hi * x_hi + y_lo * y_lo]"
# end
# let w = Constraint[]
#   @test is_free4(TracedLeaf(w, "x", -1.5), TracedLeaf(w, "y", -1.5))
#   norm_constraints = map(normalize_constraint, w)
#   @test repr(map(expr -> interval_eval(expr).hi, norm_constraints)) == "TracedSub[1.0 - x_hi * x_hi + y_hi * y_hi]"
# end
# let w = Constraint[]
#   @test is_free4(TracedLeaf(w, "x", 1.25), TracedLeaf(w, "y", -1.75))
#   norm_constraints = map(normalize_constraint, w)
#   @test repr(map(expr -> interval_eval(expr).hi, norm_constraints)) == "TracedSub[1.0 - x_lo * x_lo + y_hi * y_hi]"
# end
# let w = Constraint[]
#   # Test if x = 0, at which point the constraint could involve y_lo or y_hi,
#   # both are fair game.
#   @test is_free4(TracedLeaf(w, "x", 0), TracedLeaf(w, "y", 1.5))
#   norm_constraints = map(normalize_constraint, w)
#   @test repr(map(expr -> interval_eval(expr).hi, norm_constraints)) == "TracedSub[1.0 - x_lo * x_lo + y_lo * y_lo]"
# end

quotify(_, expr::TracedConstant) = expr.value
quotify(env, expr::TracedLeaf) = env[expr.name]
# quotify(_, expr::TracedConstant) = :($(expr.value))
# quotify(env, expr::TracedLeaf) = :($(env[expr.name]))
quotify(env, expr::TracedMul) = :($(quotify(env, expr.lhs)) * $(quotify(env, expr.rhs)))
quotify(env, expr::TracedAdd) = :($(quotify(env, expr.lhs)) + $(quotify(env, expr.rhs)))
quotify(env, expr::TracedSub) = :($(quotify(env, expr.lhs)) - $(quotify(env, expr.rhs)))

# w1 are constraints in terms of original variables, x and y. w2 are constraints
# in terms of lo/hi values, eg. x_lo, y_hi.
function poo()
let w1 = Constraint[], w2 = Constraint[]
  x_init = 0.1
  y_init = 1.1
  @test is_free4(TracedLeaf(w1, "x", x_init), TracedLeaf(w1, "y", y_init))
  foo1 = map(c -> interval_eval(w2, normalize_constraint(c)).hi, w1)
  foo2 = map(c -> normalize_constraint(c), w2)
  # @show foo1
  # @show foo2

  # model = JuMP.Model(Ipopt.Optimizer)

  model = JuMP.Model(NLopt.Optimizer)
  # JuMP.set_optimizer_attribute(model, "algorithm", :LD_MMA)
  # JuMP.set_optimizer_attribute(model, "algorithm", :LD_CCSAQ)
  JuMP.set_optimizer_attribute(model, "algorithm", :LD_SLSQP)

  vars = Dict()
  lohis = Dict()
  for (var, val) in [("x", x_init), ("y", y_init)]
    # min <= lo <= x <= hi <= max
    lo = JuMP.@variable(model, base_name=var * "_lo", lower_bound=-2.0, upper_bound=val, start=val - 1e-3)
    hi = JuMP.@variable(model, base_name=var * "_hi", lower_bound=val, upper_bound=2.0, start=val + 1e-3)
    # lo = JuMP.@variable(model, base_name=var * "_lo", lower_bound=-2.0, upper_bound=val)
    # hi = JuMP.@variable(model, base_name=var * "_hi", lower_bound=val, upper_bound=2.0)
    # @constraint(model, lo <= val)
    # @constraint(model, val <= hi)
    vars[var * "_lo"] = lo
    vars[var * "_hi"] = hi
    lohis[var] = (lo, hi)
  end
  for expr in [foo1; foo2]
    q = quotify(vars, expr)
    JuMP.add_NL_constraint(model, :($(q) <= 0))
  end
  widths = [:($(hi) - $(lo)) for (lo, hi) in values(lohis)]
  JuMP.set_NL_objective(model, MAX_SENSE, Expr(:call, :*, widths...))

  JuMP.optimize!(model)
  # @show JuMP.primal_status(model)
  # @show Dict(x => JuMP.value(v) for (x, v) in vars)
  # @show solution_summary(model, verbose = true)
  # @show model
end
end

@time poo()
@time poo()
@time poo()
@time poo()
@time poo()
