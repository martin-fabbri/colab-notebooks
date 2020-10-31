### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ acf67f82-1a85-11eb-2bdf-abc05bfb68eb
begin
	using DataFrames
	using Statistics
	using CSV
	using FreqTables
	using Pipe
end

# ╔═╡ 8061b6ba-1a85-11eb-1a73-130f152eb8d8
VERSION

# ╔═╡ ad304f82-1a85-11eb-1b1e-bba3c3d49c45
isfile.(["Project.toml", "Manifest.toml"])

# ╔═╡ 4356bc2c-1b0c-11eb-069b-155c8cf6512e
ENV["LINES"], ENV["COLUMNS"] = 15, 200

# ╔═╡ 8849a66c-1adc-11eb-3c30-371f47ba053e
md"""
# Working with groups of rows
"""

# ╔═╡ 23c9e8ac-1b0c-11eb-0834-1bdf38850a22
df = CSV.File("auto2.csv") |> DataFrame

# ╔═╡ 55f5d282-1b0c-11eb-2664-f9739e646807
md"""
We want to group our data frame by :brand
"""

# ╔═╡ 23aeac54-1b0c-11eb-3dbb-4566edc7b0e8
gdf = groupby(df, :brand)

# ╔═╡ 238fbd4c-1b0c-11eb-0d12-2b9942b58740
gdf[("ford",)]

# ╔═╡ 2334882c-1b0c-11eb-39d6-51ac9fa55c4f
md"""
Using the combine function we can easily calculate some aggregates by group:
"""

# ╔═╡ fce8af84-1b0b-11eb-0c3f-f17c2350aa0d
brand_mpg = combine(gdf, :mpg => mean)

# ╔═╡ fccdd632-1b0b-11eb-0ca3-19c005a0bc47
combine(gdf, :mpg => mean => :mean_mpg)

# ╔═╡ fcb3d11a-1b0b-11eb-0e42-8fdc9591d38e
sort!(brand_mpg, :mpg_mean, rev=true)

# ╔═╡ fc976412-1b0b-11eb-3718-39db026e228e
md"""
A typical data cleaning task is to check its consistency. In this case it would be making sure that each brand has a unique origin. We will go it several ways.
"""

# ╔═╡ fc7d9802-1b0b-11eb-20ce-15226fd3a6f5
freqtable(df, :brand, :origin)

# ╔═╡ 1d627fae-1b0e-11eb-36ce-a1d67dd1e3ad
orig_brand = @pipe df |>
				   groupby(_, :brand) |>
                   combine(_, :origin => x -> length(unique(x))) 

# ╔═╡ 1d7e8a3c-1b0e-11eb-3dfc-7fdf2376b161
extrema(orig_brand.origin_function)

# ╔═╡ 1d9a9f74-1b0e-11eb-3c64-e79a33c15318
origin_brand2 = @pipe df |>
                      groupby(_, [:origin, :brand]) |>
                      combine(_, nrow)

# ╔═╡ 1db60408-1b0e-11eb-0098-7dab96566a24
origin_vs_brand = unstack(origin_brand2, :brand, :origin, :nrow)

# ╔═╡ 1dd04eb2-1b0e-11eb-1636-3ba9f2ee7752
coalesce.(origin_vs_brand, 0)

# ╔═╡ 1debc2bc-1b0e-11eb-3be3-d981c8e2e6a5
names(origin_vs_brand)

# ╔═╡ 1e05f80a-1b0e-11eb-10b7-cfb0c4e0cf8b
propertynames(origin_vs_brand)

# ╔═╡ fc62b794-1b0b-11eb-0a82-f3cadd0ffb1c
origin_vs_brand3 = @pipe df |>
                         groupby(_, :origin) |>
                         combine(_, :brand => x -> Ref(unique(x)))

# ╔═╡ fc450492-1b0b-11eb-1257-1f35afd6ba2f
@pipe df |> groupby(_, :origin) |> combine(_, :brand => unique)

# ╔═╡ fc26545c-1b0b-11eb-025a-93aca39ca5fb
flatten(origin_vs_brand3, :brand_function)

# ╔═╡ fc091054-1b0b-11eb-00f2-4b26f065152b


# ╔═╡ fbea2a7c-1b0b-11eb-3dfd-d1edcb81816f


# ╔═╡ c5110f36-1aff-11eb-2712-e56fcfd44535


# ╔═╡ c4f62900-1aff-11eb-319c-6fbed55179ff


# ╔═╡ a79492ea-1ac8-11eb-098c-f12b9d38f313


# ╔═╡ a76f13ea-1ac8-11eb-3800-b1c27e74a466


# ╔═╡ a76f037a-1ac8-11eb-199c-cb332db6cca1


# ╔═╡ a7282752-1ac8-11eb-0f05-83cace66d3b0


# ╔═╡ ab7c9592-1a85-11eb-342f-3dab4c1d54a6


# ╔═╡ Cell order:
# ╠═8061b6ba-1a85-11eb-1a73-130f152eb8d8
# ╠═ad304f82-1a85-11eb-1b1e-bba3c3d49c45
# ╠═acf67f82-1a85-11eb-2bdf-abc05bfb68eb
# ╠═4356bc2c-1b0c-11eb-069b-155c8cf6512e
# ╟─8849a66c-1adc-11eb-3c30-371f47ba053e
# ╠═23c9e8ac-1b0c-11eb-0834-1bdf38850a22
# ╟─55f5d282-1b0c-11eb-2664-f9739e646807
# ╠═23aeac54-1b0c-11eb-3dbb-4566edc7b0e8
# ╠═238fbd4c-1b0c-11eb-0d12-2b9942b58740
# ╟─2334882c-1b0c-11eb-39d6-51ac9fa55c4f
# ╠═fce8af84-1b0b-11eb-0c3f-f17c2350aa0d
# ╠═fccdd632-1b0b-11eb-0ca3-19c005a0bc47
# ╠═fcb3d11a-1b0b-11eb-0e42-8fdc9591d38e
# ╟─fc976412-1b0b-11eb-3718-39db026e228e
# ╠═fc7d9802-1b0b-11eb-20ce-15226fd3a6f5
# ╠═1d627fae-1b0e-11eb-36ce-a1d67dd1e3ad
# ╠═1d7e8a3c-1b0e-11eb-3dfc-7fdf2376b161
# ╠═1d9a9f74-1b0e-11eb-3c64-e79a33c15318
# ╠═1db60408-1b0e-11eb-0098-7dab96566a24
# ╠═1dd04eb2-1b0e-11eb-1636-3ba9f2ee7752
# ╠═1debc2bc-1b0e-11eb-3be3-d981c8e2e6a5
# ╠═1e05f80a-1b0e-11eb-10b7-cfb0c4e0cf8b
# ╠═fc62b794-1b0b-11eb-0a82-f3cadd0ffb1c
# ╠═fc450492-1b0b-11eb-1257-1f35afd6ba2f
# ╠═fc26545c-1b0b-11eb-025a-93aca39ca5fb
# ╠═fc091054-1b0b-11eb-00f2-4b26f065152b
# ╠═fbea2a7c-1b0b-11eb-3dfd-d1edcb81816f
# ╠═c5110f36-1aff-11eb-2712-e56fcfd44535
# ╠═c4f62900-1aff-11eb-319c-6fbed55179ff
# ╠═a79492ea-1ac8-11eb-098c-f12b9d38f313
# ╠═a76f13ea-1ac8-11eb-3800-b1c27e74a466
# ╠═a76f037a-1ac8-11eb-199c-cb332db6cca1
# ╠═a7282752-1ac8-11eb-0f05-83cace66d3b0
# ╠═ab7c9592-1a85-11eb-342f-3dab4c1d54a6
