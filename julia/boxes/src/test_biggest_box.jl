include("biggest_box.jl")

import Test: @test

# Example is_free functions...
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

# TODO: these tests are out of date
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

# TODO: test the maze example with [0.9240324194846694, 4.999788161150534].
