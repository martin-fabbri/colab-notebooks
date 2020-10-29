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
typeof(w), typeof(z)

# ╔═╡ 62c234a2-19b2-11eb-357e-51f15ad594f7
md"""
## Nicer syntax for views
"""

# ╔═╡ 61d29104-19b2-11eb-02fe-4bac18bb7c78
z2 = @view v[3:5]

# ╔═╡ 3b7d1600-19b2-11eb-1fcc-af0792b5f3ea
typeof(z2)

# ╔═╡ 3b9a61c4-19b2-11eb-01b3-a3d54b983b27
z2 .= 1888

# ╔═╡ 3bdb17fc-19b2-11eb-0611-9db04a1299a0
v

# ╔═╡ 3bf85cc0-19b2-11eb-3a6b-15570698ac18
md"""
## Matrices: slices and views
"""

# ╔═╡ 3c11fa04-19b2-11eb-0065-91f886ef2445
M = [10i + j for i in 0:4, j in 1:4]

# ╔═╡ 3c2a697c-19b2-11eb-31fd-3d8d978ac8ab
M2 = [10i + j for i in 0:4 for j in 1:4]

# ╔═╡ 3c467bda-19b2-11eb-3900-1f7a1c9c723d
M[3:4, 1:2]

# ╔═╡ 3c63cbd6-19b2-11eb-3b2c-c72cf3fd8d2f
view(M, 3:5, 1:2)

# ╔═╡ 3c7753b8-19b2-11eb-200d-7b7b2c0358c7
@view M[3:4, 1:2]

# ╔═╡ 3c90fe26-19b2-11eb-2f44-256156778b3e
md"""
## Reshaping matrices
"""

# ╔═╡ 5385781c-19b4-11eb-1c42-c312f8e12946
M4 = reshape(M, 2, 10)

# ╔═╡ 53a3df6e-19b4-11eb-1de1-632c4c6d90a3
vv = vec(M)

# ╔═╡ 53bf61d0-19b4-11eb-2212-d91065dae28f


# ╔═╡ 53d99a46-19b4-11eb-21fa-859486e6289b


# ╔═╡ 53f647ea-19b4-11eb-2356-755781371a3a


# ╔═╡ 5413928c-19b4-11eb-1890-230a8077f4ae


# ╔═╡ 542f1244-19b4-11eb-207c-131b5699ad24


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
# ╟─62c234a2-19b2-11eb-357e-51f15ad594f7
# ╠═61d29104-19b2-11eb-02fe-4bac18bb7c78
# ╠═3b7d1600-19b2-11eb-1fcc-af0792b5f3ea
# ╠═3b9a61c4-19b2-11eb-01b3-a3d54b983b27
# ╠═3bdb17fc-19b2-11eb-0611-9db04a1299a0
# ╟─3bf85cc0-19b2-11eb-3a6b-15570698ac18
# ╠═3c11fa04-19b2-11eb-0065-91f886ef2445
# ╠═3c2a697c-19b2-11eb-31fd-3d8d978ac8ab
# ╠═3c467bda-19b2-11eb-3900-1f7a1c9c723d
# ╠═3c63cbd6-19b2-11eb-3b2c-c72cf3fd8d2f
# ╠═3c7753b8-19b2-11eb-200d-7b7b2c0358c7
# ╟─3c90fe26-19b2-11eb-2f44-256156778b3e
# ╠═5385781c-19b4-11eb-1c42-c312f8e12946
# ╠═53a3df6e-19b4-11eb-1de1-632c4c6d90a3
# ╠═53bf61d0-19b4-11eb-2212-d91065dae28f
# ╠═53d99a46-19b4-11eb-21fa-859486e6289b
# ╠═53f647ea-19b4-11eb-2356-755781371a3a
# ╠═5413928c-19b4-11eb-1890-230a8077f4ae
# ╠═542f1244-19b4-11eb-207c-131b5699ad24
