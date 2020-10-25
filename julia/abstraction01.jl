### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 3df9d1c0-164c-11eb-335f-edb4366d2625
using Images

# ╔═╡ 41d37ef0-164b-11eb-0940-636023edbe0d
md"""
## A concrete first taste of abstraction
"""

# ╔═╡ 7b93882c-9ad8-11ea-0288-0941e163f9d5
element = "one"

# ╔═╡ c0a59f10-164b-11eb-0936-fd3454efb65c


# ╔═╡ 3a641d50-164b-11eb-2078-b97293ae313b
fill(element, 3, 4)

# ╔═╡ 628e18d0-164b-11eb-3769-83aa55a3ed01
typeof(element)

# ╔═╡ 69747450-164b-11eb-3b17-fde3f3596580
keeptrack = [typeof(1), typeof(1.2), typeof(element), typeof(1//1), typeof([1 2; 3 4])]

# ╔═╡ 915c84d0-164b-11eb-271c-d5a01efdac33
typeof(keeptrack)

# ╔═╡ 4f0e04e0-1647-11eb-0d85-fd2fc760b466
begin
	download("https://lh3.googleusercontent.com/xw3UWSm-moxMK2G5ShMnlCZopEVwkdB6a5gl1Yy1t78bX5nGXis9cbMvHyCUG_Ahdg", "one.jpg")
	cute_one = load("one.jpg")
end

# ╔═╡ 960a8710-164c-11eb-1904-1b8980d547c5
v = [1, 2, 3, 4]

# ╔═╡ efb17020-1675-11eb-1fb7-336a4c1c2533
size(v)

# ╔═╡ f547cc00-1675-11eb-187c-e30ef94fa837
w = [ 1 2 3
      4 5 6 ]

# ╔═╡ 06e476c0-1676-11eb-0a76-1b95e30d3d34
size(w)

# ╔═╡ 0a3639d0-1676-11eb-3081-e712ab31bade
A1 = rand(1:9, 3, 4)

# ╔═╡ 205c1630-1676-11eb-3ead-df5c3fef5376
w[1, 1]

# ╔═╡ 3701d0f0-1676-11eb-2a4e-c13d2d8f9bcd
w[1, :]

# ╔═╡ 3e1a38f0-1676-11eb-11f1-fb5b84ba8bbe
w[:, 1]

# ╔═╡ 44bd9800-1676-11eb-0ff3-03fec137eb21
w[:, 2:3]

# ╔═╡ c6aae560-1677-11eb-1366-bf4c05ff16e0
function pretty(M::Matrix{T} where T<:String)
	max_lenght = maximum(length.(M))
	dv = "<div style='display:flex;flex-direction:row'>"
	HTML(dv*join([join("<div style='width:40px; text-align:center'>".*M[i,:].*"</div>", " ") for i in 1:size(M, 1)], "</div>$dv")*"</div>")
end

# ╔═╡ 4d428850-1676-11eb-39bc-0b80f8e9f835
emoji_matrix = string.(rand("🏆🏀🍇🍀🍐🍕🍩🍺🏉🐒🐘🐪🐵🐷", 3, 4)) |> pretty

# ╔═╡ b4533d20-1679-11eb-098c-e78f9ffda8e3
colors = distinguishable_colors(10)

# ╔═╡ eee808d0-1679-11eb-2b37-ddaf371d6d42
A3 = rand(colors, 10, 10)

# ╔═╡ f6ea3490-1679-11eb-123e-55037f38fdd0
D = [i*j for i=1:5, j=1:5]

# ╔═╡ 4a62c420-167a-11eb-1a12-5f1a3daef6dd
D^2

# ╔═╡ 750ba250-167a-11eb-19b0-2dc7ce04734b
D.^2

# ╔═╡ 841f3130-167a-11eb-15a5-a56c46415860
[A3 A3]

# ╔═╡ 9db7ed30-167a-11eb-3f94-ff34b57080a9
[A3 ; A3]

# ╔═╡ aa3eb6b0-167a-11eb-3048-dfd589e52566


# ╔═╡ eb45dbd0-1679-11eb-2cd8-771e19ba7c8f


# ╔═╡ e2873de0-1679-11eb-036e-8190dc0d5fc6


# ╔═╡ ac1c7ec0-1677-11eb-3ccc-4d621572f3a3


# ╔═╡ 9fd7c752-1677-11eb-3473-3d01fe6dfa38


# ╔═╡ 7f0bd6e0-164c-11eb-2f22-d375f1ac1b71


# ╔═╡ 60fef7e0-164c-11eb-1a27-0fa917856cd2


# ╔═╡ 911299a0-1642-11eb-218c-8f6840b887ab


# ╔═╡ fc23af62-15f5-11eb-0299-9bf620ca0ff1


# ╔═╡ b85681e0-15f5-11eb-14ab-c3cfbe62b4ad


# ╔═╡ e73dc280-15f4-11eb-195e-713a69ac9fbc


# ╔═╡ 3ef8fa40-15f4-11eb-1c95-672b781ada4a


# ╔═╡ 3b9d2510-15f4-11eb-1770-8d398ff60073


# ╔═╡ 2f0f7dc0-15f4-11eb-2518-59bef4e15fe2


# ╔═╡ 260530d0-15f4-11eb-2693-85f048f60c3f


# ╔═╡ 1818bcce-15f4-11eb-23bf-2b737bb9c447


# ╔═╡ 13fdb652-15f4-11eb-311f-b190620c8bd3


# ╔═╡ 10170640-15f4-11eb-294b-2740fb779cd3


# ╔═╡ 07d9310e-15f4-11eb-06d4-9141cb006dc1


# ╔═╡ e3ec42fe-15f3-11eb-1994-d9690c0e2b80


# ╔═╡ dc306f10-15f3-11eb-3799-e7eb265a5c55


# ╔═╡ d76d8fd0-15f3-11eb-21fe-bffcf0020255


# ╔═╡ d20a2ad2-15f3-11eb-370d-d56acfe3ca54


# ╔═╡ c6228780-15f3-11eb-0025-f30f5d3e89f7


# ╔═╡ b88a7510-15f3-11eb-13c0-0db9ee1d5769


# ╔═╡ 8fc4aec0-15f3-11eb-236f-0fc69d96e2ac


# ╔═╡ 8453c210-15f3-11eb-006e-c37bbd9a96e0


# ╔═╡ 7d9030d0-15f3-11eb-08fd-eda5b3de05fb


# ╔═╡ e9e32ade-15f2-11eb-23e6-8f4b9cc624da


# ╔═╡ Cell order:
# ╟─41d37ef0-164b-11eb-0940-636023edbe0d
# ╠═3df9d1c0-164c-11eb-335f-edb4366d2625
# ╠═7b93882c-9ad8-11ea-0288-0941e163f9d5
# ╠═c0a59f10-164b-11eb-0936-fd3454efb65c
# ╠═3a641d50-164b-11eb-2078-b97293ae313b
# ╠═628e18d0-164b-11eb-3769-83aa55a3ed01
# ╠═69747450-164b-11eb-3b17-fde3f3596580
# ╠═915c84d0-164b-11eb-271c-d5a01efdac33
# ╠═4f0e04e0-1647-11eb-0d85-fd2fc760b466
# ╠═960a8710-164c-11eb-1904-1b8980d547c5
# ╠═efb17020-1675-11eb-1fb7-336a4c1c2533
# ╠═f547cc00-1675-11eb-187c-e30ef94fa837
# ╠═06e476c0-1676-11eb-0a76-1b95e30d3d34
# ╠═0a3639d0-1676-11eb-3081-e712ab31bade
# ╠═205c1630-1676-11eb-3ead-df5c3fef5376
# ╠═3701d0f0-1676-11eb-2a4e-c13d2d8f9bcd
# ╠═3e1a38f0-1676-11eb-11f1-fb5b84ba8bbe
# ╠═44bd9800-1676-11eb-0ff3-03fec137eb21
# ╠═c6aae560-1677-11eb-1366-bf4c05ff16e0
# ╠═4d428850-1676-11eb-39bc-0b80f8e9f835
# ╠═b4533d20-1679-11eb-098c-e78f9ffda8e3
# ╠═eee808d0-1679-11eb-2b37-ddaf371d6d42
# ╠═f6ea3490-1679-11eb-123e-55037f38fdd0
# ╠═4a62c420-167a-11eb-1a12-5f1a3daef6dd
# ╠═750ba250-167a-11eb-19b0-2dc7ce04734b
# ╠═841f3130-167a-11eb-15a5-a56c46415860
# ╠═9db7ed30-167a-11eb-3f94-ff34b57080a9
# ╠═aa3eb6b0-167a-11eb-3048-dfd589e52566
# ╠═eb45dbd0-1679-11eb-2cd8-771e19ba7c8f
# ╠═e2873de0-1679-11eb-036e-8190dc0d5fc6
# ╠═ac1c7ec0-1677-11eb-3ccc-4d621572f3a3
# ╠═9fd7c752-1677-11eb-3473-3d01fe6dfa38
# ╠═7f0bd6e0-164c-11eb-2f22-d375f1ac1b71
# ╠═60fef7e0-164c-11eb-1a27-0fa917856cd2
# ╠═911299a0-1642-11eb-218c-8f6840b887ab
# ╠═fc23af62-15f5-11eb-0299-9bf620ca0ff1
# ╠═b85681e0-15f5-11eb-14ab-c3cfbe62b4ad
# ╠═e73dc280-15f4-11eb-195e-713a69ac9fbc
# ╠═3ef8fa40-15f4-11eb-1c95-672b781ada4a
# ╠═3b9d2510-15f4-11eb-1770-8d398ff60073
# ╠═2f0f7dc0-15f4-11eb-2518-59bef4e15fe2
# ╠═260530d0-15f4-11eb-2693-85f048f60c3f
# ╠═1818bcce-15f4-11eb-23bf-2b737bb9c447
# ╠═13fdb652-15f4-11eb-311f-b190620c8bd3
# ╠═10170640-15f4-11eb-294b-2740fb779cd3
# ╠═07d9310e-15f4-11eb-06d4-9141cb006dc1
# ╠═e3ec42fe-15f3-11eb-1994-d9690c0e2b80
# ╠═dc306f10-15f3-11eb-3799-e7eb265a5c55
# ╠═d76d8fd0-15f3-11eb-21fe-bffcf0020255
# ╠═d20a2ad2-15f3-11eb-370d-d56acfe3ca54
# ╠═c6228780-15f3-11eb-0025-f30f5d3e89f7
# ╠═b88a7510-15f3-11eb-13c0-0db9ee1d5769
# ╠═8fc4aec0-15f3-11eb-236f-0fc69d96e2ac
# ╠═8453c210-15f3-11eb-006e-c37bbd9a96e0
# ╠═7d9030d0-15f3-11eb-08fd-eda5b3de05fb
# ╠═e9e32ade-15f2-11eb-23e6-8f4b9cc624da
