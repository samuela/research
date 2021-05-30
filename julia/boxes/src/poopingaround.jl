### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 469290df-1eaf-4a96-aae9-9ab91a0dc995
using JuMP

# ╔═╡ 0e810a9e-9df2-4649-81ae-9d1ae833c5cf
using GLPK

# ╔═╡ c2610903-5e6f-4e09-a493-736c622a49b5
model = Model(GLPK.Optimizer)

# ╔═╡ b9d9cff7-7cdd-4980-8dc3-07ba3d2a549a
begin
	@variable(model, x >= 0)
	@variable(model, 0 <= y <= 3)
	@objective(model, Min, 12x + 20y)
	@constraint(model, c1, 6x + 8y >= 100)
	@constraint(model, c2, 7x + 12y >= 120)
end

# ╔═╡ 473c8b3b-b67b-4898-a803-3f180d73629a
begin
	print(model)
	optimize!(model)
	@show termination_status(model)
	@show primal_status(model)
	@show dual_status(model)
	@show objective_value(model)
	@show value(x)
	@show value(y)
	@show shadow_price(c1)
	@show shadow_price(c2)
end

# ╔═╡ d1b2d36a-3921-4760-b377-1f3d6e5ebd23


# ╔═╡ 152f9f85-8a3c-4aaf-9acd-eacf0ebadc16


# ╔═╡ bb00dfc7-7384-4fd0-8b73-2dcc2b92e0b6


# ╔═╡ Cell order:
# ╠═469290df-1eaf-4a96-aae9-9ab91a0dc995
# ╠═0e810a9e-9df2-4649-81ae-9d1ae833c5cf
# ╠═c2610903-5e6f-4e09-a493-736c622a49b5
# ╠═b9d9cff7-7cdd-4980-8dc3-07ba3d2a549a
# ╠═473c8b3b-b67b-4898-a803-3f180d73629a
# ╠═d1b2d36a-3921-4760-b377-1f3d6e5ebd23
# ╠═152f9f85-8a3c-4aaf-9acd-eacf0ebadc16
# ╠═bb00dfc7-7384-4fd0-8b73-2dcc2b92e0b6
