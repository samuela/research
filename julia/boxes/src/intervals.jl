import Base: +, -, *, ==, <=, <, big, convert, isfinite, isless, iszero, show, zero
import Test: @test

struct Interval
  lo
  hi
end

# These are useful because they can be overridden for specific types.
lt_zero(x) = x < zero(x)
gt_zero(x) = x > zero(x)
lteq_zero(x) = x <= zero(x)
gteq_zero(x) = x >= zero(x)

+(a::Interval, b::Interval) = Interval(a.lo + b.lo, a.hi + b.hi)
-(a::Interval, b::Interval) = Interval(a.lo - b.hi, a.hi - b.lo)

# See https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/master/src/intervals/arithmetic.jl#L121
*(a::Interval, b::Interval) = begin
  if gteq_zero(b.lo)
    gteq_zero(a.lo) && return Interval(a.lo * b.lo, a.hi * b.hi)
    lteq_zero(a.hi) && return Interval(a.lo * b.hi, a.hi * b.lo)
    return Interval(a.lo * b.hi, a.hi * b.hi)   # zero(T) ∈ a
  elseif lteq_zero(b.hi)
    gteq_zero(a.lo) && return Interval(a.hi * b.lo, a.lo * b.hi)
    lteq_zero(a.hi) && return Interval(a.hi * b.hi, a.lo * b.lo)
    return Interval(a.hi * b.lo, a.lo * b.lo)   # zero(T) ∈ a
  else
    gt_zero(a.lo) && return Interval(a.hi * b.lo, a.hi * b.hi)
    lt_zero(a.hi) && return Interval(a.lo * b.hi, a.lo * b.lo)
    return Interval(min(a.lo * b.hi, a.hi * b.lo), max(a.lo * b.lo, a.hi * b.hi))
  end
end

# This is likely less efficient and doesn't really fix the "quadrant" problem.
# *(a::Interval, b::Interval) = begin
#   xs = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
#   return Interval(min(xs...), max(xs...))
# end

begin
  @test Interval(1.0, 2.0) + Interval(3.0, 4.0) == Interval(4.0, 6.0)
end
