### A Pluto.jl notebook ###
# v0.12.6

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
df.x1

# ╔═╡ ac168440-1a85-11eb-3b9f-3595b29077cc
df."y1"

# ╔═╡ abf8d044-1a85-11eb-1df4-3f834e5990b9
df[:, :y1]

# ╔═╡ abdc5e8c-1a85-11eb-3b39-b500d21dd246
df[:, "y1"]

# ╔═╡ aa0e0f54-1ac8-11eb-1085-93fe0cedd3c0
select(df, r"x", :)

# ╔═╡ dcaa992a-1acb-11eb-0015-cdb2cfe62b8d
df

# ╔═╡ a9ee21f8-1ac8-11eb-0555-377010925b9a
select(df, r"x")

# ╔═╡ a9b716d6-1ac8-11eb-0ded-bdad1bea310d
describe(df, :mean => mean, :std => std)

# ╔═╡ a99835e0-1ac8-11eb-2a27-0157951b99fd
describe(df, "mean" => mean, "std" => std)

# ╔═╡ a97963ea-1ac8-11eb-0ee7-7bbf209a59fd
describe(df)

# ╔═╡ a95f9456-1ac8-11eb-14b4-c5aa281000bf
df.id = 1:nrow(df)

# ╔═╡ a91faa3a-1ac8-11eb-0d22-31a5e566a332
nrow(df)

# ╔═╡ a901e5fc-1ac8-11eb-1eb3-a11857b4bdc0
ncol(df)

# ╔═╡ a8e4981e-1ac8-11eb-0c16-07fe13d9de24
select(df, "id", :)

# ╔═╡ a8c5f24c-1ac8-11eb-387a-07909ec37e8a
df

# ╔═╡ a8a989c2-1ac8-11eb-2512-799ec5c564bf
select!(df, "id", :)

# ╔═╡ 51020740-1aca-11eb-0c90-57bad44609d8
xlim = collect(extrema(Matrix(select(df, r"x"))) .+ (-1, 1))

# ╔═╡ a84fad9c-1ac8-11eb-2492-1f456aa4f48b
ylim = collect(extrema(Matrix(select(df, r"y"))) .+ (-1, 1))

# ╔═╡ a8120980-1ac8-11eb-2b9a-69b28da0cc07
collect(10:10:100)

# ╔═╡ 853a316c-1ad2-11eb-30fa-dbc9f5e61c71
begin
	fig, axs = plt.subplots(2, 2)
	fig.tight_layout(pad=4.0)
	for i in 1:4
		x = Symbol("x", i)
		y = Symbol("y", i)
		model = lm(term(y)~term(x), df)
		axs[i].plot(xlim, predict(model, DataFrame(x => xlim)), color="orange")
		axs[i].scatter(df[:, x], df[:, y])
		axs[i].set_xlim(xlim)
		axs[i].set_ylim(ylim)
		axs[i].set_xlabel("x$i")
		axs[i].set_ylabel("y$i")
		a, b = round.(coef(model), digits=2)
		c = round(100 * r2(model), digits=2)
		axs[i].set_title(string("R²=$c%, $y=$a+$b$x"))
	end
	fig
end

# ╔═╡ 85848898-1ad2-11eb-1fa5-713858ae4565
begin
	x = :var1
	y = :var2
	
	xc = 1:3
	yc = 4:6
	
	DataFrame(x => xc, y => yc)
	
end

# ╔═╡ 85a6e21a-1ad2-11eb-00a4-cb349b78caec
DataFrame(var1=xc, var2=yc)

# ╔═╡ 85dc1928-1ad2-11eb-242b-2917e65f2244
df.x1

# ╔═╡ 85f131a0-1ad2-11eb-23ad-df576e2e6b7a


# ╔═╡ 8609a440-1ad2-11eb-0696-09f4c5a8fc0f


# ╔═╡ 8623433e-1ad2-11eb-2589-79e06429f720


# ╔═╡ a7f25360-1ac8-11eb-1c1f-552f8c1a9d8a


# ╔═╡ a7d388c2-1ac8-11eb-1bd7-59a0654fd402


# ╔═╡ a79492ea-1ac8-11eb-098c-f12b9d38f313


# ╔═╡ a76f13ea-1ac8-11eb-3800-b1c27e74a466


# ╔═╡ a76f037a-1ac8-11eb-199c-cb332db6cca1


# ╔═╡ a7282752-1ac8-11eb-0f05-83cace66d3b0


# ╔═╡ ab7c9592-1a85-11eb-342f-3dab4c1d54a6


# ╔═╡ Cell order:
# ╠═8061b6ba-1a85-11eb-1a73-130f152eb8d8
# ╠═ad304f82-1a85-11eb-1b1e-bba3c3d49c45
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
# ╠═aa0e0f54-1ac8-11eb-1085-93fe0cedd3c0
# ╠═dcaa992a-1acb-11eb-0015-cdb2cfe62b8d
# ╠═a9ee21f8-1ac8-11eb-0555-377010925b9a
# ╠═a9b716d6-1ac8-11eb-0ded-bdad1bea310d
# ╠═a99835e0-1ac8-11eb-2a27-0157951b99fd
# ╠═a97963ea-1ac8-11eb-0ee7-7bbf209a59fd
# ╠═a95f9456-1ac8-11eb-14b4-c5aa281000bf
# ╠═a91faa3a-1ac8-11eb-0d22-31a5e566a332
# ╠═a901e5fc-1ac8-11eb-1eb3-a11857b4bdc0
# ╠═a8e4981e-1ac8-11eb-0c16-07fe13d9de24
# ╠═a8c5f24c-1ac8-11eb-387a-07909ec37e8a
# ╠═a8a989c2-1ac8-11eb-2512-799ec5c564bf
# ╠═51020740-1aca-11eb-0c90-57bad44609d8
# ╠═a84fad9c-1ac8-11eb-2492-1f456aa4f48b
# ╠═a8120980-1ac8-11eb-2b9a-69b28da0cc07
# ╠═853a316c-1ad2-11eb-30fa-dbc9f5e61c71
# ╠═85848898-1ad2-11eb-1fa5-713858ae4565
# ╠═85a6e21a-1ad2-11eb-00a4-cb349b78caec
# ╠═85dc1928-1ad2-11eb-242b-2917e65f2244
# ╠═85f131a0-1ad2-11eb-23ad-df576e2e6b7a
# ╠═8609a440-1ad2-11eb-0696-09f4c5a8fc0f
# ╠═8623433e-1ad2-11eb-2589-79e06429f720
# ╠═a7f25360-1ac8-11eb-1c1f-552f8c1a9d8a
# ╠═a7d388c2-1ac8-11eb-1bd7-59a0654fd402
# ╠═a79492ea-1ac8-11eb-098c-f12b9d38f313
# ╠═a76f13ea-1ac8-11eb-3800-b1c27e74a466
# ╠═a76f037a-1ac8-11eb-199c-cb332db6cca1
# ╠═a7282752-1ac8-11eb-0f05-83cace66d3b0
# ╠═ab7c9592-1a85-11eb-342f-3dab4c1d54a6
