import Test: @test

struct Interval
  lo
  hi

  Interval(x) = new(x, x)
  Interval(lo, hi) = new(lo, hi)
end

# These are useful because they can be overridden for specific types.
lt_zero(x) = x < zero(x)
gt_zero(x) = x > zero(x)
lteq_zero(x) = x <= zero(x)
gteq_zero(x) = x >= zero(x)

Base.one(a::Interval) = Interval(one(a.lo), one(a.hi))
Base.:+(a::Interval, b::Interval) = Interval(a.lo + b.lo, a.hi + b.hi)
Base.:-(a::Interval, b::Interval) = Interval(a.lo - b.hi, a.hi - b.lo)

# See https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/master/src/intervals/arithmetic.jl#L121
Base.:*(a::Interval, b::Interval) = begin
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

# There are only two cases: interval straddles zero, or it doesn't.
Base.inv(x::Interval) =
  if x.lo <= 0 <= x.hi
    Interval(-∞, ∞)
  else
    Interval(inv(x.hi), inv(x.lo))
  end

Base.:/(a::Interval, b::Interval) = a * inv(b)

# This is likely less efficient and doesn't really fix the "quadrant" problem.
# *(a::Interval, b::Interval) = begin
#   xs = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
#   return Interval(min(xs...), max(xs...))
# end

function Base.cos(x::Interval)
	if x.hi - x.lo >= 2π
		return Interval(-1.0, 1.0)
	end

	# Using RoundNearest means wrap x.lo into the range [-π, π].
	lo = rem2pi(x.lo, RoundNearest)
	# -π <= lo <= hi <= 3π
	hi = lo + (x.hi - x.lo)

	if lo <= 0.0
		if hi <= 0.0
			# -π <= lo <= hi <= 0
			Interval(cos(lo), cos(hi))
		elseif hi <= π
			# -π <= lo <= 0 < hi <= π
			Interval(min(cos(lo), cos(hi)), 1.0)
		else
			# -π <= lo <= 0, π < hi <= 2π
			Interval(-1.0, 1.0)
		end
	else
    # TracedExpr's only support comparison
		if hi <= π
			# 0 < lo <= hi <= π
			Interval(cos(hi), cos(lo))
		elseif hi <= 2π
			# 0 < lo <= π < hi <= 2π
			Interval(-1.0, max(cos(lo), cos(hi)))
		else
			# 0 < lo <= π, 2π < hi <= 3π
			Interval(-1.0, 1.0)
		end
	end
end

Base.sin(x::Interval) = cos(Interval(0.5*π*one(x.lo)) - x)

# Square root is monotonic.
Base.sqrt(x::Interval) = Interval(sqrt(x.lo), sqrt(x.hi))

# Tests...
begin
  tuplify(x) = (x.lo, x.hi)

  @test Interval(1.0, 2.0) + Interval(3.0, 4.0) == Interval(4.0, 6.0)

  # -π <= lo <= hi <= 0
  @test tuplify(cos(Interval(-1, -0.5))) == (cos(-1), cos(-0.5))

  # -π <= lo <= 0 < hi <= π
  # cos(lo) < cos(hi)
  @test tuplify(cos(Interval(-1, 0.5))) == (cos(-1), 1)

  # -π <= lo <= 0 < hi <= π
  # cos(hi) < cos(lo)
  @test tuplify(cos(Interval(-0.5, 1))) == (cos(1), 1)

  # -π <= lo <= 0, π < hi <= 2π
  @test tuplify(cos(Interval(-1, 4))) == (-1, 1)

  @test tuplify(cos(Interval(1, 2))) == (cos(2), cos(1))
  @test tuplify(cos(Interval(1, 4))) == (-1, cos(1))
  @test tuplify(cos(Interval(1, 3.15))) == (-1, cos(1))

  @test tuplify(cos(Interval(1, 10))) == (-1, 1)

  @test convert(Interval, Interval(1, 2)) == Interval(1, 2)
  @test tuplify(Interval(2.0) * Interval(1.0, 2.0)) == (2, 4)

  # Apparently cos(0.5*π) is not approximately 0 in Julia.
  @test sin(Interval(0)).lo ≈ cos(0.5*π)
end
