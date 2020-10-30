### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ acf67f82-1a85-11eb-2bdf-abc05bfb68eb
begin
	using DataFrames
	using Statistics
	using PyPlot
	using GLM
end

# ╔═╡ 8061b6ba-1a85-11eb-1a73-130f152eb8d8
VERSION

# ╔═╡ ad304f82-1a85-11eb-1b1e-bba3c3d49c45
isfile.(["Project.toml", "Manifest.toml"])

# ╔═╡ ad14112a-1a85-11eb-2137-eda93315eb44
] status

# ╔═╡ acdb9e6a-1a85-11eb-2e56-b91f3f2c3e24
aq = [10.0   8.04  10.0  9.14  10.0   7.46   8.0   6.58
       8.0   6.95   8.0  8.14   8.0   6.77   8.0   5.76
      13.0   7.58  13.0  8.74  13.0  12.74   8.0   7.71
       9.0   8.81   9.0  8.77   9.0   7.11   8.0   8.84
      11.0   8.33  11.0  9.26  11.0   7.81   8.0   8.47
      14.0   9.96  14.0  8.1   14.0   8.84   8.0   7.04
       6.0   7.24   6.0  6.13   6.0   6.08   8.0   5.25
       4.0   4.26   4.0  3.1    4.0   5.39  19.0  12.50 
      12.0  10.84  12.0  9.13  12.0   8.15   8.0   5.56
       7.0   4.82   7.0  7.26   7.0   6.42   8.0   7.91
       5.0   5.68   5.0  4.74   5.0   5.73   8.0   6.89];

# ╔═╡ acbde3e8-1a85-11eb-2d9f-b7754fbf0281
df = DataFrame(aq)

# ╔═╡ aca17b68-1a85-11eb-39fd-bf8a9cfe42bb
newnames = vec(string.(["x", "y"], [1 2 3 4]))

# ╔═╡ ac6908d2-1a85-11eb-0733-63fcf56fecd3
rename!(df, newnames)

# ╔═╡ ac4c9b20-1a85-11eb-22d4-cb4290f63002
DataFrame(aq, [:x1, :y1, :x2, :y2, :x3, :y3, :x4, :y4])

# ╔═╡ ac31974e-1a85-11eb-1d08-f1198ac82594


# ╔═╡ ac168440-1a85-11eb-3b9f-3595b29077cc


# ╔═╡ abf8d044-1a85-11eb-1df4-3f834e5990b9


# ╔═╡ abdc5e8c-1a85-11eb-3b39-b500d21dd246


# ╔═╡ ab7c9592-1a85-11eb-342f-3dab4c1d54a6


# ╔═╡ Cell order:
# ╠═8061b6ba-1a85-11eb-1a73-130f152eb8d8
# ╠═ad304f82-1a85-11eb-1b1e-bba3c3d49c45
# ╠═ad14112a-1a85-11eb-2137-eda93315eb44
# ╠═acf67f82-1a85-11eb-2bdf-abc05bfb68eb
# ╠═acdb9e6a-1a85-11eb-2e56-b91f3f2c3e24
# ╟─acbde3e8-1a85-11eb-2d9f-b7754fbf0281
# ╟─aca17b68-1a85-11eb-39fd-bf8a9cfe42bb
# ╠═ac6908d2-1a85-11eb-0733-63fcf56fecd3
# ╠═ac4c9b20-1a85-11eb-22d4-cb4290f63002
# ╠═ac31974e-1a85-11eb-1d08-f1198ac82594
# ╠═ac168440-1a85-11eb-3b9f-3595b29077cc
# ╠═abf8d044-1a85-11eb-1df4-3f834e5990b9
# ╠═abdc5e8c-1a85-11eb-3b39-b500d21dd246
# ╠═ab7c9592-1a85-11eb-342f-3dab4c1d54a6
