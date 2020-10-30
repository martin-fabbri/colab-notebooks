### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ acf67f82-1a85-11eb-2bdf-abc05bfb68eb
begin
	using DataFrames
	using Statistics
	using CSV	
end

# ╔═╡ 8061b6ba-1a85-11eb-1a73-130f152eb8d8
VERSION

# ╔═╡ ad304f82-1a85-11eb-1b1e-bba3c3d49c45
isfile.(["Project.toml", "Manifest.toml"])

# ╔═╡ 8849a66c-1adc-11eb-3c30-371f47ba053e
md"""
# Working with text files
"""

# ╔═╡ 8609a440-1ad2-11eb-0696-09f4c5a8fc0f
begin
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original"
	download(url, "auto.txt")
end

# ╔═╡ 8623433e-1ad2-11eb-2589-79e06429f720
readlines("auto.txt")

# ╔═╡ 2b3bd64c-1add-11eb-12a7-dda4b1d43b59
md"""
## Data Cleaning
- Table has no headers
- Last column is tab separated, while earlier columns are separated with varying number of spaces
- Missing values are encoded by "NA"
"""

# ╔═╡ a7f25360-1ac8-11eb-1c1f-552f8c1a9d8a
raw_str = read("auto.txt", String)

# ╔═╡ 96c5bac2-1add-11eb-3852-4d3153fccd80
md"""
Replace all tabs with spaces
"""

# ╔═╡ 95d8036a-1add-11eb-35f4-770cc195ab45
str_no_tab = replace(raw_str, '\t' => ' ')

# ╔═╡ be35efd4-1add-11eb-1652-8bb45ea07ab1
io = IOBuffer(str_no_tab)

# ╔═╡ 6c91e6f4-1add-11eb-30b8-0711ba80a788
md"""
We can think of IO as n in-memory I/O stream. Therefore, we can pass this stream to CSV. While loading our data we are are making the following assumptions.

1. Use spacers as delimeters.
2. Ignore repeated (consecutive) occurrences of the delimiter.
3. We explicitly pass column names via header keyword argument.
4. We specify that missing values are represented using "NA"
"""

# ╔═╡ 0fc39f28-1b01-11eb-172c-c73f51a11353
sin(1.3)

# ╔═╡ 1622caa8-1b01-11eb-2c94-6ddeb98ee29e
1.2 |> sin

# ╔═╡ 6cb070c4-1add-11eb-22c4-91ccd53e92f6
df1 = CSV.File(
	io,
	delim=' ',
	ignorerepeated=true,
	header=[:mpg, :cylinders, :displacement, :horsepower,
            :weight, :acceleration, :year, :origin, :name],
	missingstring="NA") |> 
	DataFrame

# ╔═╡ 6ccb47d2-1add-11eb-20f2-a5043f12c160
df1

# ╔═╡ 6ce76124-1add-11eb-29dc-af32921df293
ENV["COLUMNS"], ENV["LINES"] = 200, 15

# ╔═╡ 6d02368e-1add-11eb-252d-bdb11b1c3e41
df1

# ╔═╡ 6d1d1652-1add-11eb-0f14-e541320df058
df_raw = CSV.File("auto.txt", header=[:metrics, :name]) |> DataFrame

# ╔═╡ 6d37f116-1add-11eb-1a4a-fbd3da64ef4c
str_metrics = split.(df_raw.metrics)

# ╔═╡ fc665566-1afc-11eb-1bf5-ebff28a03169
df1_2 = DataFrame(fill(Float64, 8), [:mpg, :cylinders, :displacement, :horsepower, :weight, :acceleration, :year, :origin])

# ╔═╡ fc4dba10-1afc-11eb-32b6-e5eff96c251f
allowmissing!(df1_2, [:mpg, :horsepower])

# ╔═╡ fc3001fa-1afc-11eb-3f59-a779618967c0
for row in str_metrics
	push!(df1_2, [v == "NA" ? missing : parse(Float64, v) for v in row])
end

# ╔═╡ fc1655ca-1afc-11eb-1a6d-2de5383414fd
df1_2

# ╔═╡ a30237b4-1afd-11eb-2a46-4dd62d19b907
df1_2.name = df_raw.name

# ╔═╡ fbfa3032-1afc-11eb-2f21-9f3afc8a66ef
df1_2

# ╔═╡ fbdedf8c-1afc-11eb-33f2-49a07fb38b1e
df1_2.name == df_raw.name

# ╔═╡ fbbeeeaa-1afc-11eb-0cb9-bd01a64ef604
df1_2.name = df_raw[:, :name]

# ╔═╡ fba1329a-1afc-11eb-37a0-55342cda66c9
isequal(df1_2, df1)

# ╔═╡ 6d540338-1add-11eb-0442-67bbdfeba0fb
sum(count(ismissing, col) for col in eachcol(df1))

# ╔═╡ 6d701668-1add-11eb-1494-8ddd04b56334
mapcols(x -> count(ismissing, x), df1)

# ╔═╡ 6ddb83bc-1add-11eb-1e8c-752a464a1ba4
filter(row -> any(ismissing, row), df1)

# ╔═╡ 6df796f6-1add-11eb-2cf8-557ff3745852
df1.brand = first.(split.(df1.name))

# ╔═╡ 6e1b0dca-1add-11eb-1d1a-23b292fe2b3c
df2 = dropmissing(df1)

# ╔═╡ a7d388c2-1ac8-11eb-1bd7-59a0654fd402
df2[df2.brand .== "saab", :]

# ╔═╡ c5824836-1aff-11eb-21cf-2d5e2c4c58b9
filter(:brand => ==("saab"), df2)

# ╔═╡ c565d67e-1aff-11eb-1e6d-2d137f119d62
filter(row -> row.brand == "saab", df2)

# ╔═╡ c54c47a4-1aff-11eb-15d4-97cbec4614d5
CSV.write("auto2.csv", df2)

# ╔═╡ c52fda08-1aff-11eb-1635-7769f5e1fb29
readlines("auto2.csv")

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
# ╟─8849a66c-1adc-11eb-3c30-371f47ba053e
# ╠═8609a440-1ad2-11eb-0696-09f4c5a8fc0f
# ╠═8623433e-1ad2-11eb-2589-79e06429f720
# ╟─2b3bd64c-1add-11eb-12a7-dda4b1d43b59
# ╠═a7f25360-1ac8-11eb-1c1f-552f8c1a9d8a
# ╟─96c5bac2-1add-11eb-3852-4d3153fccd80
# ╠═95d8036a-1add-11eb-35f4-770cc195ab45
# ╠═be35efd4-1add-11eb-1652-8bb45ea07ab1
# ╟─6c91e6f4-1add-11eb-30b8-0711ba80a788
# ╠═0fc39f28-1b01-11eb-172c-c73f51a11353
# ╠═1622caa8-1b01-11eb-2c94-6ddeb98ee29e
# ╠═6cb070c4-1add-11eb-22c4-91ccd53e92f6
# ╠═6ccb47d2-1add-11eb-20f2-a5043f12c160
# ╠═6ce76124-1add-11eb-29dc-af32921df293
# ╠═6d02368e-1add-11eb-252d-bdb11b1c3e41
# ╠═6d1d1652-1add-11eb-0f14-e541320df058
# ╠═6d37f116-1add-11eb-1a4a-fbd3da64ef4c
# ╠═fc665566-1afc-11eb-1bf5-ebff28a03169
# ╠═fc4dba10-1afc-11eb-32b6-e5eff96c251f
# ╠═fc3001fa-1afc-11eb-3f59-a779618967c0
# ╠═fc1655ca-1afc-11eb-1a6d-2de5383414fd
# ╠═a30237b4-1afd-11eb-2a46-4dd62d19b907
# ╠═fbfa3032-1afc-11eb-2f21-9f3afc8a66ef
# ╠═fbdedf8c-1afc-11eb-33f2-49a07fb38b1e
# ╠═fbbeeeaa-1afc-11eb-0cb9-bd01a64ef604
# ╠═fba1329a-1afc-11eb-37a0-55342cda66c9
# ╠═6d540338-1add-11eb-0442-67bbdfeba0fb
# ╠═6d701668-1add-11eb-1494-8ddd04b56334
# ╠═6ddb83bc-1add-11eb-1e8c-752a464a1ba4
# ╠═6df796f6-1add-11eb-2cf8-557ff3745852
# ╠═6e1b0dca-1add-11eb-1d1a-23b292fe2b3c
# ╠═a7d388c2-1ac8-11eb-1bd7-59a0654fd402
# ╠═c5824836-1aff-11eb-21cf-2d5e2c4c58b9
# ╠═c565d67e-1aff-11eb-1e6d-2d137f119d62
# ╠═c54c47a4-1aff-11eb-15d4-97cbec4614d5
# ╠═c52fda08-1aff-11eb-1635-7769f5e1fb29
# ╠═c5110f36-1aff-11eb-2712-e56fcfd44535
# ╠═c4f62900-1aff-11eb-319c-6fbed55179ff
# ╠═a79492ea-1ac8-11eb-098c-f12b9d38f313
# ╠═a76f13ea-1ac8-11eb-3800-b1c27e74a466
# ╠═a76f037a-1ac8-11eb-199c-cb332db6cca1
# ╠═a7282752-1ac8-11eb-0f05-83cace66d3b0
# ╠═ab7c9592-1a85-11eb-342f-3dab4c1d54a6
