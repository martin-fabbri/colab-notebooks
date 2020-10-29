### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ b2a8e540-19af-11eb-0214-9d5f04ac2486
md"""
# Arrays: Slices and views
"""

# ╔═╡ 3df9d1c0-164c-11eb-335f-edb4366d2625
v = ones(Int, 10)

# ╔═╡ 78a80050-173c-11eb-2fe8-6b70bb6a2507
v[3]

# ╔═╡ 06176472-19b0-11eb-21ec-0744703214b6
v[3] = 2

# ╔═╡ 1c31ef66-19b0-11eb-3c7b-87e5b3b253ac
v[3:5]

# ╔═╡ 1b45c1cc-19b0-11eb-04d2-937a0b44df8f
v[3:5] = 4

# ╔═╡ 1b274daa-19b0-11eb-3d39-63379843e8f5
v[3:5] .= 4

# ╔═╡ 1b099b70-19b0-11eb-2cd2-b535b3df5c37
v

# ╔═╡ 05bc6914-19b0-11eb-0857-1320b5d269fc
v[5:7] = [100, 101, 102]

# ╔═╡ 0572949c-19b0-11eb-3861-8bb389451ba7
v

# ╔═╡ 051ef344-19b0-11eb-0215-1b67608a6e35
w = v[3:5]

# ╔═╡ e9c25c40-19b0-11eb-0e7d-5b8230a3990f
w[1] = 444

# ╔═╡ e9a11bd4-19b0-11eb-385e-4f08b8c3e34d
w

# ╔═╡ e97fd756-19b0-11eb-395d-4fd72a6b4998
v

# ╔═╡ e961047c-19b0-11eb-37f5-bdcb55b8c5bd
z = view(v, 3:5)

# ╔═╡ e94283e6-19b0-11eb-2fe0-c1799fe420af
z

# ╔═╡ e923aa8c-19b0-11eb-03b4-63258474e7d8
z[1] = 555

# ╔═╡ e903b1fa-19b0-11eb-2f85-99101781fcac
z

# ╔═╡ e8e52762-19b0-11eb-11f4-d373924283e3
v

# ╔═╡ e8c15eea-19b0-11eb-018e-eda8a769bebb
z .= 9

# ╔═╡ e87f1062-19b0-11eb-3a5e-d764456e1a0e
v

# ╔═╡ e85d9e00-19b0-11eb-3925-99918cddcbd4


# ╔═╡ Cell order:
# ╟─b2a8e540-19af-11eb-0214-9d5f04ac2486
# ╠═3df9d1c0-164c-11eb-335f-edb4366d2625
# ╠═78a80050-173c-11eb-2fe8-6b70bb6a2507
# ╠═06176472-19b0-11eb-21ec-0744703214b6
# ╠═1c31ef66-19b0-11eb-3c7b-87e5b3b253ac
# ╠═1b45c1cc-19b0-11eb-04d2-937a0b44df8f
# ╠═1b274daa-19b0-11eb-3d39-63379843e8f5
# ╠═1b099b70-19b0-11eb-2cd2-b535b3df5c37
# ╠═05bc6914-19b0-11eb-0857-1320b5d269fc
# ╠═0572949c-19b0-11eb-3861-8bb389451ba7
# ╠═051ef344-19b0-11eb-0215-1b67608a6e35
# ╠═e9c25c40-19b0-11eb-0e7d-5b8230a3990f
# ╠═e9a11bd4-19b0-11eb-385e-4f08b8c3e34d
# ╠═e97fd756-19b0-11eb-395d-4fd72a6b4998
# ╠═e961047c-19b0-11eb-37f5-bdcb55b8c5bd
# ╠═e94283e6-19b0-11eb-2fe0-c1799fe420af
# ╠═e923aa8c-19b0-11eb-03b4-63258474e7d8
# ╠═e903b1fa-19b0-11eb-2f85-99101781fcac
# ╠═e8e52762-19b0-11eb-11f4-d373924283e3
# ╠═e8c15eea-19b0-11eb-018e-eda8a769bebb
# ╠═e87f1062-19b0-11eb-3a5e-d764456e1a0e
# ╠═e85d9e00-19b0-11eb-3925-99918cddcbd4
