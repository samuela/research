### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 9cb7277a-1ea5-49b3-872c-db9c9ae715dc
include("ingredients.jl")

# ╔═╡ fef90971-2b5d-4ee0-a7cb-f91e46a06ebb
import PyCall: pyimport

# ╔═╡ 8b855b0c-e8f9-4837-acf4-ec9f811822e3
import PyPlot

# ╔═╡ d17094f7-ac5d-4af3-86b6-2ab12ea343f6
import Mazes

# ╔═╡ 9aacbfd1-4c63-4a4b-ae8a-16864365d6f3
import Random

# ╔═╡ a7b68252-a3c6-4d29-848e-a87fecf4a934
import SimpleGraphs: Grid, has

# ╔═╡ 64a87d28-a448-45b6-99c1-0306678f6dff
import Test: @test

# ╔═╡ d3d20a71-aaea-48a3-9991-55d363b7a97e
patches = pyimport("matplotlib.patches")

# ╔═╡ bed16984-da71-44ef-808d-06195b62c942
begin
	_mod = ingredients("boxes.jl")
	import ._mod: Box, BoxSet, in_box, add_box!, rand_inside

	_mod = ingredients("biggest_box.jl")
	import ._mod: biggest_box_2d
end

# ╔═╡ 02efeda6-1b36-4d60-a681-08bfc3722f75
function _group_contiguous_seqs(xs)
  if length(xs) == 0
    []
  else
    seqs = [[xs[1]]]
    for (i, j) in zip(xs, xs[2:end])
      if i + 1 == j
        push!(seqs[end], j)
      else
        push!(seqs, [j])
      end
    end
    seqs
  end
end

# ╔═╡ d330a755-e368-4f24-8095-c18e1a755605
@test _group_contiguous_seqs([]) == []

# ╔═╡ 586bd72b-5d4c-4d8b-a48f-ed58a9064807
@test _group_contiguous_seqs([1, 2, 4, 5, 6]) == [[1, 2], [4, 5, 6]]

# ╔═╡ 49455f3c-49b5-445c-b092-2462517a1af3
@test _group_contiguous_seqs([1, 2, 4, 5, 7]) == [[1, 2], [4, 5], [7]]

# ╔═╡ f66a8a1c-d5f6-4c85-ab1c-0bdb068d412c
@test _group_contiguous_seqs([1, 2, 3]) == [[1, 2, 3]]

# ╔═╡ 90eac080-a1bc-4889-bdb4-257b39122f51
@test _group_contiguous_seqs([1, 3, 5]) == [[1], [3], [5]]

# ╔═╡ 4d4fb0f8-dafb-4509-af9a-8b346b285100
begin
	Random.seed!(123)
	maze = Mazes.Maze(5, 7)
	bounds = Box([0, 0], [maze.c, maze.r])
end

# ╔═╡ 9ae7c0d9-40e5-4dcc-9524-33f6d6fb3783
function _puzzle_to_boxes(M::Mazes.Maze)
  G = Grid(M.r, M.c)

  vert_walls = [[] for _ in 1:M.c]
  horz_walls = [[] for _ in 1:M.r]

	non_edge_list = [e for e in G.E if !has(M.T, e[1], e[2])]
	for (a, b) in non_edge_list
		if a[1] == b[1]
			# @show a[2], b[2]
  			push!(vert_walls[min(a[2], b[2])], a[1])
		else
			@assert a[2] == b[2]
			# @show a[1], b[1]
			push!(horz_walls[min(a[1], b[1])], a[2])
		end
	end

	ϵ = 0.1
	boxes = []
	for (y, seqs) in enumerate([_group_contiguous_seqs(sort(xs)) for xs in vert_walls])
		for xs in seqs
			# Note that ys[1] = min(ys) and ys[end] = max(ys)
			push!(boxes, Box([y - ϵ, M.r - xs[end] - ϵ], [y + ϵ, M.r - xs[1] + 1 + ϵ]))
		end
	end
	for (x, seqs) in enumerate([_group_contiguous_seqs(sort(ys)) for ys in horz_walls])
		for ys in seqs
			push!(boxes, Box([ys[1] - 1 - ϵ, M.r - x - ϵ], [ys[end] + ϵ, M.r - x + ϵ]))
		end
	end

	# Add boundaries
	ϵ = 0.2
	r, c = maze.r, maze.c
	push!(boxes, Box([-ϵ, -ϵ], [c + ϵ, 0]))
	push!(boxes, Box([-ϵ, -ϵ], [0, r + ϵ]))
	push!(boxes, Box([-ϵ, r], [c + ϵ, r + ϵ]))
	push!(boxes, Box([c, -ϵ], [c + ϵ, r + ϵ]))

	boxes
end

# ╔═╡ 359c53c5-0834-4870-8e97-6cf1af53e1d3
# These are all of the "edges" that are actually walls.
[e for e in Grid(maze.r, maze.c).E if !has(maze.T, e[1], e[2])]

# ╔═╡ 3f382913-3e41-4a18-8409-f180f06c6c92
wall_boxes = _puzzle_to_boxes(maze)

# ╔═╡ 7ad04f4a-cb3e-4cef-9483-fe4b41a248bc
Mazes.draw(maze)

# ╔═╡ 619e142a-3265-470b-9b07-e155ed88ddd6
function plot_box(ax, box; kwargs...)
	w, h = box.maxs - box.mins
	ax.add_patch(patches.Rectangle(box.mins, w, h; kwargs...))
end

# ╔═╡ cf593cd7-bb3c-4dfa-9ed6-97a6741c1789
function plot_maze()
	fig, ax = PyPlot.subplots()
	for box in wall_boxes
		plot_box(ax, box; color = :darkgrey)
	end
	PyPlot.axis("equal")
	PyPlot.xticks([])
	PyPlot.yticks([])
	fig, ax
end

# ╔═╡ 05da4d08-292b-423b-9fa7-266a67daf81d
function is_free_naive(x, y)
	!any(in_box(b, [x, y]) for b in wall_boxes)
end

# ╔═╡ c0ec67e3-8a3f-4d73-bc98-fd7a8c6e6416
begin
	x_boxes = [Box([b.mins[1]], [b.maxs[1]]) for b in wall_boxes]
	y_boxes = [Box([b.mins[2]], [b.maxs[2]]) for b in wall_boxes]
	
	function is_free_fancy(x, y)
		if rand(Bool)
			x_in = findall([in_box(b, [x]) for b in x_boxes])
			!any(in_box(b, [y]) for b in y_boxes[x_in])
		else
			y_in = findall([in_box(b, [y]) for b in y_boxes])
			!any(in_box(b, [x]) for b in x_boxes[y_in])
		end
	end
end

# ╔═╡ 1247d7ed-abf4-4bd0-ac38-e657da065b39
rand(Bool)

# ╔═╡ 85dca7ad-28be-49f6-bbe2-38535f0ca06f
begin
	_, _ax = plot_maze()
	Random.seed!(123)
	# x, y = rand_inside(bounds)
	# x, y = 6.45, 0.45
	# TODO: this is a naughty point. Write a test for it.
	x, y = [0.9240324194846694, 4.999788161150534]
	@show _, box = biggest_box_2d(is_free_naive, x, y, bounds)

	PyPlot.scatter([x], [y], s = 10, zorder = 2)
	plot_box(_ax, box, color = :lightblue, alpha = 0.5)

	PyPlot.gcf()
end

# ╔═╡ 017bc257-ade5-40f0-abe5-b6b12334ef7c
results_dir = mkpath("/tmp/results/pluto_maze")

# ╔═╡ 70e09838-02ac-4c9d-a081-c5429968a6e2
begin
	Random.seed!(123)

	free_boxset = BoxSet(Box[], Dict())

	function sample_point()
		for _ in 1:1000
			x, y = rand_inside(bounds)
			if !any(in_box.(free_boxset.boxes, Ref([x, y])))
				# Not in any existing box yet: try evaluating...
				free, box = biggest_box_2d(is_free_fancy, x, y, bounds)
				if free
					return [x, y], box
				end
			end
		end
		error("oopsie: couldn't reasonably find free point")
	end
	
	figs = map(1:27) do i
		@show i
		@time xy, box = sample_point()
		# This is a stupid hack to get around pluto include module issues...
		add_box!(free_boxset, Box(box.mins, box.maxs))

		_, _ax = plot_maze()
		PyPlot.scatter([xy[1]], [xy[2]], s = 10, zorder = 2)
		for b in free_boxset.boxes
			plot_box(_ax, b, color = :lightblue, alpha = 0.5)
		end
		
		# .. to get up from the src directory. pdf's for 
		PyPlot.savefig("$results_dir/step$i.pdf")
		PyPlot.savefig("$results_dir/step$i.jpg")
		PyPlot.gcf()
	end

	# -r: framerate
	run(`ffmpeg -y -r 2 -i $results_dir/step%d.jpg $results_dir/maze.gif`)

	figs
end

# ╔═╡ Cell order:
# ╠═fef90971-2b5d-4ee0-a7cb-f91e46a06ebb
# ╠═8b855b0c-e8f9-4837-acf4-ec9f811822e3
# ╠═d17094f7-ac5d-4af3-86b6-2ab12ea343f6
# ╠═9aacbfd1-4c63-4a4b-ae8a-16864365d6f3
# ╠═a7b68252-a3c6-4d29-848e-a87fecf4a934
# ╠═64a87d28-a448-45b6-99c1-0306678f6dff
# ╠═d3d20a71-aaea-48a3-9991-55d363b7a97e
# ╠═9cb7277a-1ea5-49b3-872c-db9c9ae715dc
# ╠═bed16984-da71-44ef-808d-06195b62c942
# ╟─02efeda6-1b36-4d60-a681-08bfc3722f75
# ╟─d330a755-e368-4f24-8095-c18e1a755605
# ╟─586bd72b-5d4c-4d8b-a48f-ed58a9064807
# ╟─49455f3c-49b5-445c-b092-2462517a1af3
# ╟─f66a8a1c-d5f6-4c85-ab1c-0bdb068d412c
# ╟─90eac080-a1bc-4889-bdb4-257b39122f51
# ╟─9ae7c0d9-40e5-4dcc-9524-33f6d6fb3783
# ╠═4d4fb0f8-dafb-4509-af9a-8b346b285100
# ╠═359c53c5-0834-4870-8e97-6cf1af53e1d3
# ╠═3f382913-3e41-4a18-8409-f180f06c6c92
# ╠═7ad04f4a-cb3e-4cef-9483-fe4b41a248bc
# ╟─619e142a-3265-470b-9b07-e155ed88ddd6
# ╟─cf593cd7-bb3c-4dfa-9ed6-97a6741c1789
# ╠═05da4d08-292b-423b-9fa7-266a67daf81d
# ╠═c0ec67e3-8a3f-4d73-bc98-fd7a8c6e6416
# ╠═1247d7ed-abf4-4bd0-ac38-e657da065b39
# ╠═85dca7ad-28be-49f6-bbe2-38535f0ca06f
# ╠═017bc257-ade5-40f0-abe5-b6b12334ef7c
# ╠═70e09838-02ac-4c9d-a081-c5429968a6e2
