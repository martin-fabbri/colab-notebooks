### A Pluto.jl notebook ###
# v0.12.4

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

# ╔═╡ 39b0d07a-1a70-11eb-2379-f7db05b0309f
begin
	using Plots, PlutoUI
	import CSV, DataFrames, Dates
end

# ╔═╡ 71c4521a-1a6f-11eb-248f-2b3c6a1fd63c
md"""
# Covid Data Exploration
"""

# ╔═╡ 7729bea0-1a71-11eb-1bcd-35dbb24fad87
begin
	url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv";
	download(url, "covid_global_cases.csv")
	data = CSV.read("covid_global_cases.csv") |> 
		df -> DataFrames.rename(df, 1 => "province", 2 => "country")
	rm("covid_global_cases.csv")
end

# ╔═╡ f3c46c64-1a6f-11eb-046f-ef04da3cc23c
begin
	date_labels = names(data)[5:end]
	date_format = Dates.DateFormat("m/d/y")
	dates = Dates.Date.(date_labels, date_format) .+ Dates.Year(2000)
end

# ╔═╡ f3a333fa-1a6f-11eb-38cf-63eb06a4e8a0
countries = unique(data[:, :country]);

# ╔═╡ f38336e0-1a6f-11eb-35d1-cd48853a70d7
data_by_country = data |>
	df -> DataFrames.groupby(df, :country) |>
	df -> DataFrames.combine(df, (date_labels .=> sum .=> date_labels));

# ╔═╡ f366cfc8-1a6f-11eb-2e5e-b77251d64915
data_by_country;

# ╔═╡ c6bc690a-1a75-11eb-0348-8f17b8ac5298
md"""
## Per Country Helper Functions
"""

# ╔═╡ f32a56b0-1a6f-11eb-3d9b-018e87833269
function make_first_plot()
	base_plot = plot()
	xlabel!(base_plot, "Dates")
	ylabel!(base_plot, "Confirmed Cases")
	title!(base_plot, "Confirmed Covid Cases")
	
	base_plot
end

# ╔═╡ 47aefd04-1a77-11eb-242b-91ca5d5e1c43
function get_country_data(country)
	data_by_country |>
		df -> filter(:country => val -> val == country, df) |>
		df -> df[1, 2:end] |>
		df -> convert(Vector, df)
end

# ╔═╡ f30b96e2-1a6f-11eb-35eb-bbbc58be9c7b
function add_plot!(target_plot, country, index)
	country_data = get_country_data(country)
	plot!(target_plot,
	 	dates[1:index], country_data[1:index],
	 	xticks    = dates[1:12:end],
	 	xrotation = 45,
	 	legend = :topleft,
	 	label = country,
	 	lw = 3)
end

# ╔═╡ 2f89793e-1a79-11eb-0324-0d0e6365b50e
@bind selected_countries MultiSelect(
	[ctry => ctry for ctry in countries],
	default = ["US", "Brazil", "China", "Italy", "India"])

# ╔═╡ 7ca306b8-1a79-11eb-3f41-df9344c279f3
@bind dates_index Slider(1:length(dates), default = 40)

# ╔═╡ f2ef88a2-1a6f-11eb-1d8f-efd4f136e8b7
begin
	output_plot = make_first_plot()
	for country in selected_countries
		add_plot!(output_plot, country, dates_index)
	end
	output_plot
end

# ╔═╡ f2d07866-1a6f-11eb-1e71-fbf0edb3e3b9
data_by_country |>
	df -> df[df.country .== "US", 2:end] |>
	df -> convert(Vector, df[1, :])

# ╔═╡ f2969d30-1a6f-11eb-377c-5db9eed4a9c6


# ╔═╡ f2790d7e-1a6f-11eb-013a-ddf620c7d40a


# ╔═╡ f25dd522-1a6f-11eb-3e31-79412281c1cb


# ╔═╡ f24049f6-1a6f-11eb-232a-cd95846ccc90


# ╔═╡ f2043556-1a6f-11eb-2ff8-3945b6356950


# ╔═╡ Cell order:
# ╟─71c4521a-1a6f-11eb-248f-2b3c6a1fd63c
# ╠═39b0d07a-1a70-11eb-2379-f7db05b0309f
# ╠═7729bea0-1a71-11eb-1bcd-35dbb24fad87
# ╠═f3c46c64-1a6f-11eb-046f-ef04da3cc23c
# ╠═f3a333fa-1a6f-11eb-38cf-63eb06a4e8a0
# ╠═f38336e0-1a6f-11eb-35d1-cd48853a70d7
# ╠═f366cfc8-1a6f-11eb-2e5e-b77251d64915
# ╟─c6bc690a-1a75-11eb-0348-8f17b8ac5298
# ╠═f32a56b0-1a6f-11eb-3d9b-018e87833269
# ╠═47aefd04-1a77-11eb-242b-91ca5d5e1c43
# ╠═f30b96e2-1a6f-11eb-35eb-bbbc58be9c7b
# ╟─2f89793e-1a79-11eb-0324-0d0e6365b50e
# ╠═7ca306b8-1a79-11eb-3f41-df9344c279f3
# ╠═f2ef88a2-1a6f-11eb-1d8f-efd4f136e8b7
# ╠═f2d07866-1a6f-11eb-1e71-fbf0edb3e3b9
# ╠═f2969d30-1a6f-11eb-377c-5db9eed4a9c6
# ╠═f2790d7e-1a6f-11eb-013a-ddf620c7d40a
# ╠═f25dd522-1a6f-11eb-3e31-79412281c1cb
# ╠═f24049f6-1a6f-11eb-232a-cd95846ccc90
# ╠═f2043556-1a6f-11eb-2ff8-3945b6356950
