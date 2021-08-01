### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 2d76b8b7-31d3-4668-af60-690ba5b7f77b
md"""
# God Give Me a Sine: Adventures in Interval Arithmetic

#### [Samuel Ainsworth](https://samlikes.pizza/)
"""

# ╔═╡ 7430b073-e62d-4822-86aa-e361ab21637c
md"""
[Interval arithmetic](https://en.wikipedia.org/wiki/Interval_arithmetic) is an extension of standard arithmetic to work on intervals of values instead of values themselves. It has applications across applied mathematics, computer science, engineering, and so on. Though oddly it doesn't receive that much love in the usual undergrad CS curriculum. Anyhow, I came across a cute little interval arithmetic puzzle recently, and I figure it'd be selfish not to share...
"""

# ╔═╡ 261afaaa-0bf4-46d5-a527-1f63619a729a
md"""
## Interval arithmetic in 2 minutes

Interval arithmetic takes some computational expression, say $f(x,y)=3x^3-y$, and asks the question: Instead of plugging in real numbers for $x$ and $y$, eg. $x=5$ and $y=\sqrt{3}$, what happens if we plug in interval values, eg. $x = [4, 6]$ and $y = [\sqrt{3},\sqrt{5}]$? In other words, let's say we don't know $x$ and $y$ exactly but we do know they must lie within some bounds. Can we efficiently bound the quantity $3x^3-y$ for all such inputs?

Well, yes, as it turns out. And that's what interval arithmetic is all about. You can start by defining standard arithmetic operators on intervals and then compose operators together.

Let's start by defining an interval,
"""

# ╔═╡ 4ddbc2ef-f31b-465c-8763-f2327e1037a9
struct Interval
	# invariant: lo <= hi
	lo
	hi
end

# ╔═╡ b7729ec2-63e4-48bf-b05b-8e99762b7d39
Interval(3, 4)

# ╔═╡ 1f1332b0-b2f7-4a10-9c05-e89988f0158c
md"""
So far so easy. Ok, let's start thinking about how to define arithmetic operations on intervals. Addition is a good place to start. Consider $[x_\ell, x_h] + [y_\ell, y_h]$. We know that the minimum possible value must be $x_\ell+y_\ell$. And correspondingly the maximum value possible is $x_h+y_h$. So,

$[x_\ell, x_h] + [y_\ell, y_h] = [x_\ell+y_\ell, x_h+y_h]$

Let's make that happen,
"""

# ╔═╡ dc8674b9-a30e-4b86-ac75-dded2ea73e11
Base.:+(x::Interval, y::Interval) = Interval(x.lo + y.lo, x.hi + y.hi)

# ╔═╡ ee7d4bfd-d171-419b-a888-91e1ec6abde0
Interval(1, 2) + Interval(1.5, 4)

# ╔═╡ a5e39d2a-39cc-4455-a229-76ccf32c9933
md"""
Another easy one to define is negation:

$-[\ell, h] = [-h, -\ell]$
"""

# ╔═╡ 367a923f-2b26-499e-8005-ab9389eddeb9
Base.:-(x::Interval) = Interval(-x.hi, -x.lo)

# ╔═╡ ae269b5c-723d-4458-afb1-4b7cd3769f42
-Interval(3, 4)

# ╔═╡ 9dcd083e-255a-4fa5-9377-3820556e57dc
md"""
And we can compose addition and negation to get subtraction:
"""

# ╔═╡ eb583ac0-04bf-44b6-b5ba-a71afde08d8e
Base.:-(x::Interval, y::Interval) = x + (-y)

# ╔═╡ 2565ae7a-446f-4ad2-982b-d582dc2d88de
Interval(1, 2) - Interval(1.5, 4)

# ╔═╡ d877e8c6-86b1-47fc-8f5e-cbc4b16a5bce
md"""
Scaling an interval by some positive number, $c > 0$, is pretty simple:

$c [\ell, h] = [c\ell, ch]$
"""

# ╔═╡ db5db0e8-2b6b-4257-9bee-59f3e1b82438
Base.:*(c::Float64, x::Interval) = (@assert c > 0; Interval(c * x.lo, c * x.hi))

# ╔═╡ 346a9af3-d401-4308-9d12-b888ff6d2acf
md"""
There's one final puzzle piece we need before we can get to the good stuff: if $f$ is monotonic, then $f([\ell, h]) = [f(\ell),f(h)]$. This follows directly from the definition of monotonicity, but it can also be handy to draw it out to convince yourself.

To recap, we now have
* addition
* negation
* subtraction
* positive scaling
* monotonic functions
all for intervals.
"""

# ╔═╡ a1727c73-2f6d-4866-ac39-0d0f7e406a67
md"""
## Our feature presentation

Ok, so we're well on our way to filling out a decent interval analysis library. It's worth noting that different interval analysis implementations prioritize different goals. For example, many interval analysis libraries focus on _verified_ numerics. They care about [rounding modes](https://en.wikipedia.org/wiki/IEEE_754#Rounding_rules) and such, but we don't have time for all that noise. Let's stick to doing things that are mathematically sound, not necessarily floating point sound.

(Fun fact: the official IEEE 754 spec for floating point numerics is paywalled. What kind of BS is that?)

First and foremost, it's important to note that our implementation is _sound_: it will never return an interval that is too small. Technically, an implementation of interval arithmetic could always just return $[-\infty,\infty]$ for any operation and still be sound. But of course that would be useless. One very nice property of everything we've implemented up to this point is that all operations result in intervals that are as small as possible. So far every operation we've implemented is sound (up to floating point weirdness), and as "specific" as possible. There's no overapproximation happening anywhere. You can verify this for yourself for each operation we've defined. Formally you might say something like "we return an interval that is a subset of all intervals that could be returned by all sound implementations." Let's call interval computations that have this property _precise_.

!!! question
	Using only the precise operations we've defined, is it possible construct a function that is not precise?

Or equivalently,

!!! question
	Given two computations -- using only precise operations -- that are mathematically equivalent, eg. $f_1(x) = x^2 + 3x + 2$ and $f_2(x) = (x+1)(x+2)$, is it possible for them to have different interval results?

(Consider pausing here if you'd like to take a stab at this yourself!)
"""

# ╔═╡ 0b554d43-62c0-4c1a-b0f9-1fc95bd9e213
md"""
To answer these questions, consider the sine function, $\sin(x)$. Recently I've been developing an interval analysis library that avoids branching (if-else, min, max, etc) at all costs. The reasons are top secret, but this is the design goal. The sine function presents a bit of an issue. Most existing implementations of sine in interval arithmetic use branching extensively. Can we do better?

Well, I thought it might be possible. Consider the [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) expansion of $\sin(x)$,

$\sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \frac{x^9}{9!} -  \dots$

What luck! We know how to do interval arithmetic for everything here, and without any branching! $f(x)=x^n$ for odd $n$ is monotonic, and "dividing" an interval by some positive number is no problem either -- it's just a scaling of the interval.

Plugging in $x=[\ell,h]$,

$$\begin{align*}
\sin([\ell,h]) &= [\ell,h] - \frac{[\ell,h]^3}{3!} + \frac{[\ell,h]^5}{5!} - \frac{[\ell,h]^7}{7!} + \frac{[\ell,h]^9}{9!} -  \dots \\
&= [\ell,h] - \frac{[\ell^3,h^3]}{3!} + \frac{[\ell^5,h^5]}{5!} - \frac{[\ell^7,h^7]}{7!} + \frac{[\ell^9,h^9]}{9!} - \dots \\
&= [\ell,h] + \frac{[-h^3,-\ell^3]}{3!} + \frac{[\ell^5,h^5]}{5!} + \frac{[-h^7,-\ell^7]}{7!} + \frac{[\ell^9,h^9]}{9!} + \dots \\
&= \left[ \ell - \frac{h^3}{3!} + \frac{\ell^5}{5!} - \frac{h^7}{7!} + \frac{\ell^9}{9!} - \dots,\quad h - \frac{\ell^3}{3!} + \frac{h^5}{5!} - \frac{\ell^7}{7!} + \frac{h^9}{9!} - \dots \right]
\end{align*}$$

Supposing we define some function

$\phi(\alpha, \beta) = \alpha - \frac{\beta^3}{3!} + \frac{\alpha^5}{5!} - \frac{\beta^7}{7!} + \frac{\alpha^9}{9!} - \dots,$

we can simplify things a bit:

$\sin([\ell,h]) = [\phi(\ell, h), \phi(h,\ell)].$

Unfortunately $\phi(\alpha, \beta)$ requires computing some infinite series as it's currently written. Is there some more convenient analytic form we could use instead? Yes, in fact:

$$\phi(\alpha, \beta) = \frac{1}{2} \left[ \sin(\alpha) + \sin(\beta) + \sinh(\alpha) - \sinh(\beta) \right]$$

Obviously, right?

Let's implement this now.
"""

# ╔═╡ f7d005e5-7975-484b-95f5-eae64d9e7eab
ϕ(α, β) = 0.5 * ( (sinh(α) + sin(α)) - (sinh(β) - sin(β)) )

# ╔═╡ 4c454a8a-915b-429f-bb30-78fb91b1e862
Base.sin(x::Interval) = Interval(ϕ(x.lo, x.hi), ϕ(x.hi, x.lo))

# ╔═╡ e41cc379-9b81-48f0-9540-cfbbcb002f88
md"""
Ok, nice. Let's try playing around with it a bit...
"""

# ╔═╡ 08206fb2-6499-409e-b981-b4879ada9b78
md"""
Great! Looks like it's working! Here we're plotting our trusty $\sin(x)$ as a blue line, the low and high points of the input interval are plotted in red, and the output interval is shown as a box on the y-axis in light blue.

Let's test another interval...
"""

# ╔═╡ 5c83fc81-642f-4841-9cea-3a08974af3d0
md"""
What??? How did we get an interval that's larger than $[-1, 1]$, the total range of $\sin(x)$? After all, we only used our "safe" operations to get to this point. There's no overapproximation of the intermediate interval computations. There's no approximation error from truncating the Taylor series. It's not a floating point arithmetic issue either.

So what's going on? Well, perhaps surprisingly, the answer to both of our original questions is **_yes_**. The composition of precise interval arithmetic operations is not necessarily precise. If you're like me this fact is initially surprising! In the interval arithmetic world, it's known as the [dependency problem](https://en.wikipedia.org/wiki/Interval_arithmetic#Dependency_problem). The root issue here is that from the perspective of interval arithmetic, intervals are all completely independent of each other even though they may be correlated in the underlying mathematical expression. This effect makes for some kind of "constructive interference" between intervals. Even if two intervals represent negatively correlated values, this cannot be known from the perspective of interval arithmetic. Also, note that intervals can only ever "grow" in some sense, since they must always over-approximate the true range of values. So any error introduced in a sub-computation necessarily snowballs through the rest.

Practically speaking it means that although interval arithmetic generally works very well without supervision, you may want to write hand-crafted interval arithmetic functions for certain operations or consider different possible rewrites of your computation. The sine implementation in (to my knowledge) all existing interval analysis libraries branches extensively and does provide a precise result.

Am I dismayed at the failure of my trick here? Yeah, it's a bummer. But it's a valuable lesson learned as well. And who knows... for someone out there maybe this implementation of sine will be useful one day. It is sound after all.
"""

# ╔═╡ 67c86106-ad23-41c9-b644-ba4da14c1f90
md"""
## Conclusion

Readers familiar interval arithmetic might point out that the canonical $f_1(x) = 0, f_2(x) = x - x$ example also proves a similar point. That's true but also kind of pointless IMHO. I've never found the $x-x$ example to be compelling. I don't know anyone who would ever write code like that. But this sine example is real. I wrote it in fact!

Missing from this post is a discussion of the plentiful examples of interval arithmetic behaving sensibly and solving all sorts of exciting real world problems. For those interested in learning more, check out:
* The [Herbie project](https://herbie.uwplse.org/) uses interval arithmetic to assist in improving the accuracy of floating point code. In particular it leverages [rival](https://github.com/herbie-fp/rival), an interval arithmetic implementation that additionally supports "boolean intervals" and tracks error information in intervals.
* [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl) is a modern implementation of interval arithmetic in Julia. It implements sine the "right" way.
* [Affine arithmetic](https://en.wikipedia.org/wiki/Affine_arithmetic) is another alternative arithmetic that addresses the dependency problem to some extent. It's not a choice without tradeoffs however!
* More broadly, interval analysis is an example of [abstract interpretation](https://en.wikipedia.org/wiki/Abstract_interpretation).

_If you liked this post check me out on [Twitter](https://twitter.com/SamuelAinsworth) or [GitHub](https://github.com/samuela). My website is [samlikes.pizza](https://samlikes.pizza/) and not [samuelainsworth.com](http://samuelainsworth.com)._
"""

# ╔═╡ e0cb9a1f-9388-4555-99f0-0c11787198fa
md"""
### Acknowledgements
Shout out to Krishna Pillutla for identifying the closed form of $\phi(\alpha,\beta)$. Thanks to Oliver Flatt, Pavel Panchekha, and Krishna for reviewing drafts of this post and insightful discussion! Big thanks to Zach Tatlock for the Thai food!
"""

# ╔═╡ d834f262-7c8e-4961-87cf-a698594d57a5
import PyPlot

# ╔═╡ 491356fb-9293-4cc8-9dbb-36d2ceb6c6e7
let fig = PyPlot.figure(figsize=(8, 4)),
	x = Interval(-0.1, 0.1),
	# xs = -2:0.01:2,
	xs = -π:0.01:π
	PyPlot.plot(xs, sin.(xs), label = "\$\\sin(x)\$")
	PyPlot.plot(x.lo:0.01:x.hi, sin.(x.lo:0.01:x.hi), color = :tomato)
	PyPlot.scatter([x.lo, x.hi], [sin(x.lo), sin(x.hi)], color = :tomato, zorder = 10, label = "\$\\sin(\\ell)\$, \$\\sin(h)\$")
	PyPlot.gca().axhspan(sin(x).lo, sin(x).hi, alpha = 0.2, label = "\$\\sin([\\ell,h])\$")
	PyPlot.gca().grid()
	PyPlot.legend(loc = "upper left")
	fig
end

# ╔═╡ b5bdc8d2-157d-420b-bd41-27baf2ccf22b
let fig = PyPlot.figure(figsize=(8, 4)),
	x = Interval(0.5, 2.25),
	# xs = -2:0.01:2,
	xs = -π:0.01:π
	PyPlot.plot(xs, sin.(xs), label = "\$\\sin(x)\$")
	PyPlot.plot(x.lo:0.01:x.hi, sin.(x.lo:0.01:x.hi), color = :tomato)
	PyPlot.scatter([x.lo, x.hi], [sin(x.lo), sin(x.hi)], color = :tomato, zorder = 10, label = "\$\\sin(\\ell)\$, \$\\sin(h)\$")
	PyPlot.gca().axhspan(sin(x).lo, sin(x).hi, alpha = 0.2, label = "\$\\sin([\\ell,h])\$")
	PyPlot.gca().grid()
	PyPlot.legend(loc = "upper left")
	fig
end

# ╔═╡ 92a7cd08-a43f-410a-8c7e-2ae49dd84741
PyPlot.rc("text", usetex=true)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"

[compat]
PyPlot = "~2.9.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "94bf17e83a0e4b20c8d77f6af8ffe8cc3b386c0a"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "169bb8ea6b1b143c5cf57df6d34d022a7b60c6db"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.3"

[[PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "67dde2482fe1a72ef62ed93f8c239f947638e5a2"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.9.0"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"
"""

# ╔═╡ Cell order:
# ╟─2d76b8b7-31d3-4668-af60-690ba5b7f77b
# ╟─7430b073-e62d-4822-86aa-e361ab21637c
# ╟─261afaaa-0bf4-46d5-a527-1f63619a729a
# ╠═4ddbc2ef-f31b-465c-8763-f2327e1037a9
# ╠═b7729ec2-63e4-48bf-b05b-8e99762b7d39
# ╟─1f1332b0-b2f7-4a10-9c05-e89988f0158c
# ╠═dc8674b9-a30e-4b86-ac75-dded2ea73e11
# ╠═ee7d4bfd-d171-419b-a888-91e1ec6abde0
# ╟─a5e39d2a-39cc-4455-a229-76ccf32c9933
# ╠═367a923f-2b26-499e-8005-ab9389eddeb9
# ╠═ae269b5c-723d-4458-afb1-4b7cd3769f42
# ╟─9dcd083e-255a-4fa5-9377-3820556e57dc
# ╠═eb583ac0-04bf-44b6-b5ba-a71afde08d8e
# ╠═2565ae7a-446f-4ad2-982b-d582dc2d88de
# ╟─d877e8c6-86b1-47fc-8f5e-cbc4b16a5bce
# ╠═db5db0e8-2b6b-4257-9bee-59f3e1b82438
# ╟─346a9af3-d401-4308-9d12-b888ff6d2acf
# ╟─a1727c73-2f6d-4866-ac39-0d0f7e406a67
# ╟─0b554d43-62c0-4c1a-b0f9-1fc95bd9e213
# ╠═f7d005e5-7975-484b-95f5-eae64d9e7eab
# ╠═4c454a8a-915b-429f-bb30-78fb91b1e862
# ╟─e41cc379-9b81-48f0-9540-cfbbcb002f88
# ╟─491356fb-9293-4cc8-9dbb-36d2ceb6c6e7
# ╟─08206fb2-6499-409e-b981-b4879ada9b78
# ╟─b5bdc8d2-157d-420b-bd41-27baf2ccf22b
# ╟─5c83fc81-642f-4841-9cea-3a08974af3d0
# ╟─67c86106-ad23-41c9-b644-ba4da14c1f90
# ╟─e0cb9a1f-9388-4555-99f0-0c11787198fa
# ╟─d834f262-7c8e-4961-87cf-a698594d57a5
# ╟─92a7cd08-a43f-410a-8c7e-2ae49dd84741
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
