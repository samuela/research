### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 42f9fa77-ebbc-4134-b26e-dbd738780aaa
include("ingredients.jl")

# ╔═╡ 0e329542-2156-4700-9eed-c30f9f0f6a29
import LinearAlgebra: dot

# ╔═╡ a1ac79c8-f4a1-11eb-23ac-234d01562127
import PyPlot

# ╔═╡ 227b91b7-9f80-437a-b1e1-ec9c4572a1ac
import PlutoUI: Slider

# ╔═╡ 8cbca069-10b1-4d1d-950a-31fcd29530c3
import PyCall: pyimport

# ╔═╡ fe03384a-d837-4ece-998c-1330f1bf4c75
import Random

# ╔═╡ e4fb7afd-1dbc-46d0-be67-16d3d6d9ac83
# Packages we don't need directly, but that are needed by scripts that we call so we have to make Pluto.jl install them.
import JuMP, MathOptInterface, NLopt

# ╔═╡ bb749a34-80a1-434c-9297-8178302131a0
begin
	PyPlot.rc("text", usetex=true)
	mpatches = pyimport("matplotlib.patches")
	mcollections = pyimport("matplotlib.collections")
end

# ╔═╡ b388c99a-d15a-4373-a08d-80e5af8b00ac
begin
	_mod = ingredients("boxes.jl")
	import ._mod: BoxSet, in_box, add_box!, rand_inside, intersect_boxes

	_mod = ingredients("biggest_box.jl")
	# Pluto file includes suck ass so we need to import Box through biggest_box.jl
	import ._mod: biggest_box_2d, TracedExpr, TracedConstant, Box
end

# ╔═╡ f94da24c-e1fd-46a2-82b4-94d912cb9655
function forward_kinematics(θ1, θ2)
	y1, x1 = sincos(θ1)
	δy2, δx2 = sincos(θ2)
	[x1, y1], [x1 + δx2, y1 + δy2]
end

# ╔═╡ 7dc8b472-d1f3-4cd8-b459-90033ab4b1b6
bounds = Box([-π, -π], [π, π])

# ╔═╡ 283344fa-be92-491e-9912-d984e15b9058
md"""
θ1: $(@bind θ1_example Slider(-π:0.01:π))

θ2: $(@bind θ2_example Slider(-π:0.01:π))
"""

# ╔═╡ b75b38d5-4940-41cd-a934-68ff199d7353
θ1_example, θ2_example

# ╔═╡ bd2c7c98-1823-401e-b0f5-fa8b1ef521de
# obstacles = [([-0.8, 0.8], 0.5), ([0.8, 1.3], 0.3), ([0.8, -1.0], 0.4), ([-1.2, -1.0], 0.3)]
obstacles = [([-2.0, -2.0], 0.5)]

# ╔═╡ 70f25364-acc1-4695-8055-493249870f35
function point_in_sphere(x, sphere)
	c, r = sphere
	# @show (sum((x-c) .* (x-c)))
	sum((x-c) .* (x-c)) <= r^2
end

# ╔═╡ 8635d8dc-a9d7-4a4c-908a-fd9a931d8a67
function segment_sphere_intersect(a, b, sphere)
	# Let AB denote the line segment and let the center of the sphere be C.
	c, r = sphere
	ab = b - a
	ac = c - a
	# The projection of AC onto AB is A + t * (B-A).
	t = sum(ab .* ac) / sqrt(sum(ab .* ab))
	# The vector from A + t * (B-A) to C
	tc = c - (a + t * ab)
	squared_dist = sum(tc .* tc)
	point_in_sphere(a, sphere) || point_in_sphere(b, sphere) || (0.0 <= t <= 1.0 && squared_dist <= r^2)
end

# ╔═╡ f174064c-aeb0-4b2f-b783-cc541b89fa41
begin
	@show x = [0.1, 0.2]
	@show y = [-0.5, 1.0]
	x - y
end

# ╔═╡ abfedd8b-00a7-44c6-a4e5-16e18db88ffe
# (-)-A-B
segment_sphere_intersect([1.0], [2.0], ([0.0], 0.1)) == false

# ╔═╡ 3b6a574d-5dc2-43e5-99bb-baf4eda4444a
# (-A-)-B
segment_sphere_intersect([1.0], [2.0], ([0.0], 1.5))

# ╔═╡ 0b8b3d8a-a449-4c2c-8c22-449eef875b14
# (-A-B-)
segment_sphere_intersect([1.0], [2.0], ([0.0], 5.0))

# ╔═╡ 97418187-abbf-46aa-a578-b4e2f2a72181
# A-(-)-B
segment_sphere_intersect([1.0], [2.0], ([1.5], 0.1))

# ╔═╡ 4166f142-bddb-46ab-b57a-94405fa332b7
# A-(-B-)
segment_sphere_intersect([1.0], [2.0], ([1.9], 0.2))

# ╔═╡ 64ac676e-9437-4189-92d0-0cfe18a66cf9
# A-B-(-)
segment_sphere_intersect([1.0], [2.0], ([3], 0.2)) == false

# ╔═╡ a453a6d3-c3c5-4f9a-9843-c20eb0665015
# 0<=t<=1 but sphere is too far away
segment_sphere_intersect([1.0, 0.0], [2.0, 0.0], ([1.5, 1.0], 0.2)) == false

# ╔═╡ bf87b567-1f7c-451c-9bb1-1e5bf8c71277
# 0<=t<=1 and sphere is close enough
segment_sphere_intersect([1.0, 0.0], [2.0, 0.0], ([1.5, 0.1], 0.2))

# ╔═╡ 1a784294-565a-4b14-93ea-24752b53129f
function is_free(θ1, θ2)
	xy1, xy2 = forward_kinematics(θ1, θ2)
	!any(segment_sphere_intersect(zero(xy1), xy1, s) for s in obstacles) && !any(segment_sphere_intersect(xy1, xy2, s) for s in obstacles)
end

# ╔═╡ 72ad1c70-c9c6-46a8-bacb-64a845928486
begin
	fig, ax = PyPlot.subplots(1, 2, figsize=(8, 4))

	ax[1].set_aspect("equal", "box")
	ax[1].set_xlim(-π, π)
	ax[1].set_ylim(-π, π)
	ax[1].scatter([θ1_example], [θ2_example])
	ax[1].set_title("Configuration space")
	ax[1].set_xlabel("\$\\theta_1\$", fontsize=16)
	ax[1].set_ylabel("\$\\theta_2\$", fontsize=16)

	xy1, xy2 = forward_kinematics(θ1_example, θ2_example)
	ax[2].set_aspect("equal", "box")
	ax[2].set_xlim(-2, 2)
	ax[2].set_ylim(-2, 2)
	ax[2].plot([0, xy1[1], xy2[1]], [0, xy1[2], xy2[2]], lw=10, color="tab:pink")
	ax[2].plot([0, xy1[1], xy2[1]], [0, xy1[2], xy2[2]], lw=2, color=:black)
	ax[2].scatter([0, xy1[1]], [0, xy1[2]], color=:black, marker="o", s=30, zorder=100)
	ax[2].add_collection(mcollections.PatchCollection([
		mpatches.Circle(xy, r, color="tab:grey")
		for (xy, r) in obstacles], match_original=true))
	ax[2].set_title("Workspace")
	ax[2].set_xlabel("\$x\$", fontsize=16)
	ax[2].set_ylabel("\$y\$", fontsize=16)
	if is_free(θ1_example, θ2_example)
		ax[2].set_title("Free")
	else
		ax[2].set_title("Collision")
	end

	PyPlot.tight_layout()
	fig
end

# ╔═╡ c585b62b-b2dd-46d2-b462-e547b4c76d4c
results_dir = mkpath("/tmp/results/pluto_reacher_2d")

# ╔═╡ aea3271a-c0ea-4d38-bb3d-5aa5265a1682
function plot_box(ax, box; kwargs...)
	w, h = box.maxs - box.mins
	ax.add_patch(mpatches.Rectangle(box.mins, w, h; kwargs...))
end

# ╔═╡ 7938ffb6-3482-445c-a2de-fca65a7a964d
begin
	Random.seed!(123)

	free_boxset = BoxSet(Box[], Dict())
	coll_boxes = Box[]
	
	overlap = 1e-3

	function sample_point()
		for _ in 1:1000
			θ1, θ2 = rand_inside(bounds)
			if !any(in_box.(free_boxset.boxes, Ref([θ1, θ2]))) && !any(in_box.(coll_boxes, Ref([θ1, θ2])))
				# Not in any existing box yet: try evaluating...
				free, box = biggest_box_2d(is_free, θ1, θ2, bounds)
				if free
					add_box!(free_boxset, Box(box.mins .- overlap, box.maxs .+ overlap))
				else
					# We don't need overlap adjustment for collision boxes.
					push!(coll_boxes, box)
				end
				
				return [θ1, θ2], free, box
			end
		end
		error("oopsie: couldn't reasonably find free point")
	end
	
	figs = map(1:1) do i
		@show i
		@show θ, free, box = sample_point()

		# Plot all the things!!!
		fig, ax = PyPlot.subplots(1, 2, figsize=(8, 4))
		PyPlot.suptitle("Iteration $i")

		ax[1].set_aspect("equal", "box")
		ax[1].set_xlim(-π, π)
		ax[1].set_ylim(-π, π)
		ax[1].set_title("Configuration space")
		ax[1].set_xlabel("\$\\theta_1\$", fontsize=16)
		ax[1].set_ylabel("\$\\theta_2\$", fontsize=16)
		ax[1].scatter([θ[1]], [θ[2]], color = :gray)
		for b in free_boxset.boxes
			plot_box(ax[1], b, color = :lightblue, alpha = 0.5)
		end
		for b in coll_boxes
			plot_box(ax[1], b, color = :tomato, alpha = 0.5)
		end

		xy1, xy2 = forward_kinematics(θ[1], θ[2])
		ax[2].set_aspect("equal", "box")
		ax[2].set_xlim(-2, 2)
		ax[2].set_ylim(-2, 2)
		ax[2].plot([0, xy1[1], xy2[1]], [0, xy1[2], xy2[2]], lw=10, color="tab:pink")
		ax[2].plot([0, xy1[1], xy2[1]], [0, xy1[2], xy2[2]], lw=2, color=:black)
		ax[2].scatter([0, xy1[1]], [0, xy1[2]], color=:black, marker="o", s=30, zorder=100)
		ax[2].add_collection(mcollections.PatchCollection([
			mpatches.Circle(xy, r, color="tab:grey")
			for (xy, r) in obstacles], match_original=true))
		ax[2].set_title("Workspace")
		ax[2].set_xlabel("\$x\$", fontsize=16)
		ax[2].set_ylabel("\$y\$", fontsize=16)
		if free
			ax[2].set_title("Free")
		else
			ax[2].set_title("Collision")
		end

		PyPlot.tight_layout()
		# .. to get up from the src directory. pdf's for stills and jpg's for the gif
		PyPlot.savefig("$results_dir/step$i.pdf")
		# PyPlot.savefig("$results_dir/step$i.jpg")
		PyPlot.gcf()
	end

	# -r: framerate
	# run(`ffmpeg -y -r 2 -i $results_dir/step%d.jpg $results_dir/reacher_2d_slow.gif`)
	# run(`ffmpeg -y -r 500 -i $results_dir/step%d.jpg $results_dir/reacher_2d_fast.gif`)
	
	figs
end

# ╔═╡ a987df5d-4d91-4420-81a0-87eb6acdcea0
inv(2.0)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
NLopt = "76087f3c-5699-56af-9a33-bf431cd00edd"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
JuMP = "~0.21.9"
MathOptInterface = "~0.9.22"
NLopt = "~0.6.3"
PlutoUI = "~0.7.9"
PyCall = "~1.92.3"
PyPlot = "~2.9.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Statistics", "UUIDs"]
git-tree-sha1 = "c31ebabde28d102b602bada60ce8922c266d205b"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.1.1"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bdc0937269321858ab2a4f288486cb258b9a0af7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.0"

[[CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

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

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "344f143fa0ec67e47917848795ab19c6a455f32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.32.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "85d2d9e2524da988bffaf2a381864e20d2dae08d"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.2.1"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "44e3b40da000eab4ccb1aecdc4801c040026aeb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.13"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[JSONSchema]]
deps = ["HTTP", "JSON", "ZipFile"]
git-tree-sha1 = "b84ab8139afde82c7c65ba2b792fe12e01dd7307"
uuid = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
version = "0.3.3"

[[JuMP]]
deps = ["Calculus", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MathOptInterface", "MutableArithmetics", "NaNMath", "Printf", "Random", "SparseArrays", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "4f0a771949bbe24bf70c89e8032c107ebe03f6ba"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "0.21.9"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "JSON", "JSONSchema", "LinearAlgebra", "MutableArithmetics", "OrderedCollections", "SparseArrays", "Test", "Unicode"]
git-tree-sha1 = "575644e3c05b258250bb599e57cf73bbf1062901"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "0.9.22"

[[MathProgBase]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9abbe463a1e9fc507f12a69e7f29346c2cdc472c"
uuid = "fdba3010-5040-5b88-9595-932c9decdf73"
version = "0.7.8"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "3927848ccebcc165952dc0d9ac9aa274a87bfe01"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.2.20"

[[NLopt]]
deps = ["MathOptInterface", "MathProgBase", "NLopt_jll"]
git-tree-sha1 = "d80cb3327d1aeef0f59eacf225e000f86e4eee0a"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "0.6.3"

[[NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "2b597c46900f5f811bec31f0dcc88b45744a2a09"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.7.0+0"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "bfd7d8c7fd87f04543810d9cbd3995972236ba1b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.2"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

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

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "508822dca004bf62e210609148511ad03ce8f1d8"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3fedeffc02e47d6e3eb479150c8e5cd8f15a77a0"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.10"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "7c53c35547de1c5b9d46a4797cf6d8253807108c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.5"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"

[[ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "c3a5637e27e914a7a445b8d0ad063d701931e9f7"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═0e329542-2156-4700-9eed-c30f9f0f6a29
# ╠═a1ac79c8-f4a1-11eb-23ac-234d01562127
# ╠═227b91b7-9f80-437a-b1e1-ec9c4572a1ac
# ╠═8cbca069-10b1-4d1d-950a-31fcd29530c3
# ╠═fe03384a-d837-4ece-998c-1330f1bf4c75
# ╠═e4fb7afd-1dbc-46d0-be67-16d3d6d9ac83
# ╠═bb749a34-80a1-434c-9297-8178302131a0
# ╠═42f9fa77-ebbc-4134-b26e-dbd738780aaa
# ╠═b388c99a-d15a-4373-a08d-80e5af8b00ac
# ╠═f94da24c-e1fd-46a2-82b4-94d912cb9655
# ╠═7dc8b472-d1f3-4cd8-b459-90033ab4b1b6
# ╟─283344fa-be92-491e-9912-d984e15b9058
# ╟─b75b38d5-4940-41cd-a934-68ff199d7353
# ╠═bd2c7c98-1823-401e-b0f5-fa8b1ef521de
# ╠═72ad1c70-c9c6-46a8-bacb-64a845928486
# ╠═70f25364-acc1-4695-8055-493249870f35
# ╠═8635d8dc-a9d7-4a4c-908a-fd9a931d8a67
# ╠═f174064c-aeb0-4b2f-b783-cc541b89fa41
# ╠═abfedd8b-00a7-44c6-a4e5-16e18db88ffe
# ╠═3b6a574d-5dc2-43e5-99bb-baf4eda4444a
# ╠═0b8b3d8a-a449-4c2c-8c22-449eef875b14
# ╠═97418187-abbf-46aa-a578-b4e2f2a72181
# ╠═4166f142-bddb-46ab-b57a-94405fa332b7
# ╠═64ac676e-9437-4189-92d0-0cfe18a66cf9
# ╠═a453a6d3-c3c5-4f9a-9843-c20eb0665015
# ╠═bf87b567-1f7c-451c-9bb1-1e5bf8c71277
# ╠═1a784294-565a-4b14-93ea-24752b53129f
# ╠═c585b62b-b2dd-46d2-b462-e547b4c76d4c
# ╠═7938ffb6-3482-445c-a2de-fca65a7a964d
# ╠═aea3271a-c0ea-4d38-bb3d-5aa5265a1682
# ╠═a987df5d-4d91-4420-81a0-87eb6acdcea0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
