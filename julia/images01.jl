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

# ╔═╡ a0f0ae42-9ad8-11ea-2475-a735df64aa28
begin
	# Pkg.add(["Images", "ImageIO", "ImageMagick"])
	using Images
end

# ╔═╡ baeb2820-1645-11eb-3e15-0fe5abc28118
using PlutoUI

# ╔═╡ 7b93882c-9ad8-11ea-0288-0941e163f9d5
md"""
# Image processing in Julia
"""

# ╔═╡ 2b356cb0-1648-11eb-3a0a-6b810440efc9


# ╔═╡ d889e13e-f0ff-11ea-3306-57491904d83f
md"_First, we set up a clean package environment:_"

# ╔═╡ aefb6004-f0ff-11ea-10c9-c504c7aa9fe5
# begin
#	import Pkg
#   Pkg.activate(mktempdir())
# end

# ╔═╡ e48a0e78-f0ff-11ea-2852-2f4eee1a7d2d
md"_Next, we add the `Images` package to our environment, and we import it._"

# ╔═╡ 5ae65950-9ad9-11ea-2e14-35119d369acd
md"""
## The basics
Now we can get started. Let's load an image and explore its properties. 
"""

# ╔═╡ aaa805d4-9ad8-11ea-21c2-3b20580fea0e
url = "https://eskipaper.com/images/shihtzu-1.jpg"

# ╔═╡ 2ac9f6c0-15f2-11eb-249d-77733d9b3fd5
puppy_file = download(url, "puppy.jpg")

# ╔═╡ 5e9abc00-15f2-11eb-08b8-31eef538d1eb
puppy = load(puppy_file)

# ╔═╡ ea52b590-15f2-11eb-2f93-1319c9d9c0e4
typeof(puppy)

# ╔═╡ f50693d0-15f2-11eb-14e8-ffcada6c1bd1
RGBX(0.4, 0.4, 0.6)

# ╔═╡ 0bcc80c0-15f3-11eb-254b-d744f1b6f701
size(puppy)

# ╔═╡ 285caad0-15f3-11eb-1dd2-496a616913f7
begin
	(h, w) = size(puppy)
	eye = puppy[(h ÷ 3): (h ÷ 2), (w ÷ 3): (w ÷ 2)]
end

# ╔═╡ 43f11a50-15f4-11eb-34f5-8b2b836cf772
size(eye)

# ╔═╡ 4b812350-15f4-11eb-1095-6f1ea5754557
[
	eye                   reverse(eye, dims=2)
]

# ╔═╡ ed5d72a0-15f4-11eb-319b-85235c633494
new_puppy = copy(puppy);

# ╔═╡ bfea3460-15f5-11eb-235f-cb0abc6406d0
red = RGB(1, 0, 0)

# ╔═╡ e01ed560-15f5-11eb-3154-9f6cc9722dee
for i in 1:100
	for j in 1:300
		new_puppy[i, j] = red
	end 
end

# ╔═╡ 0406ba60-15f6-11eb-1c26-bbfe9bd0ee3d
new_puppy

# ╔═╡ 21251ab0-15f6-11eb-3561-1fa3e9f7de1b
begin
	new_puppy2 = copy(new_puppy)
	new_puppy2[100:200, 1:100] .= RGB(0, 1, 0)
	new_puppy2
end

# ╔═╡ 8baf4632-15f6-11eb-23a9-d7afc60ae64f
function redify(color)
	return RGB(color.r, 0, 0)
end

# ╔═╡ b77fb470-15f6-11eb-10ef-fb0c8a53747b
begin
	color = RGB(0.8, 0.5, 0.2)
	[color, redify(color)]
end

# ╔═╡ dec73f32-15f6-11eb-1617-11cad6e316bf
redify.(puppy)

# ╔═╡ ea558d60-1642-11eb-0d64-9ff0b0223125
decimate(image, ratio=5) = image[1:ratio:end, 1:ratio:end]

# ╔═╡ 13bf4660-15f7-11eb-0bac-676621168238
begin
	poor_puppy = decimate(puppy, 20)
	size(poor_puppy)
	poor_puppy
end

# ╔═╡ 145bea00-1643-11eb-2a19-fbfb2f2e6072
# convolve(puppy, blur(2))

# ╔═╡ 4464d1d0-1643-11eb-1f61-a51dfd39197e


# ╔═╡ 7e2f48a0-1643-11eb-2e90-131b0e9580f5
md"""
## Experiments
"""

# ╔═╡ c8088c4e-1645-11eb-164f-e5e814cbb6ff
@bind repeat_count Slider(1:10, show_value=true)

# ╔═╡ 0eb90b00-1643-11eb-28b9-9f6f753d8861
repeat(poor_puppy, repeat_count, repeat_count)

# ╔═╡ 4f0e04e0-1647-11eb-0d85-fd2fc760b466


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
# ╟─7b93882c-9ad8-11ea-0288-0941e163f9d5
# ╠═2b356cb0-1648-11eb-3a0a-6b810440efc9
# ╟─d889e13e-f0ff-11ea-3306-57491904d83f
# ╠═aefb6004-f0ff-11ea-10c9-c504c7aa9fe5
# ╟─e48a0e78-f0ff-11ea-2852-2f4eee1a7d2d
# ╠═a0f0ae42-9ad8-11ea-2475-a735df64aa28
# ╟─5ae65950-9ad9-11ea-2e14-35119d369acd
# ╠═aaa805d4-9ad8-11ea-21c2-3b20580fea0e
# ╠═2ac9f6c0-15f2-11eb-249d-77733d9b3fd5
# ╠═5e9abc00-15f2-11eb-08b8-31eef538d1eb
# ╠═ea52b590-15f2-11eb-2f93-1319c9d9c0e4
# ╠═f50693d0-15f2-11eb-14e8-ffcada6c1bd1
# ╠═0bcc80c0-15f3-11eb-254b-d744f1b6f701
# ╠═285caad0-15f3-11eb-1dd2-496a616913f7
# ╠═43f11a50-15f4-11eb-34f5-8b2b836cf772
# ╠═4b812350-15f4-11eb-1095-6f1ea5754557
# ╠═ed5d72a0-15f4-11eb-319b-85235c633494
# ╠═bfea3460-15f5-11eb-235f-cb0abc6406d0
# ╠═e01ed560-15f5-11eb-3154-9f6cc9722dee
# ╠═0406ba60-15f6-11eb-1c26-bbfe9bd0ee3d
# ╠═21251ab0-15f6-11eb-3561-1fa3e9f7de1b
# ╠═8baf4632-15f6-11eb-23a9-d7afc60ae64f
# ╠═b77fb470-15f6-11eb-10ef-fb0c8a53747b
# ╠═dec73f32-15f6-11eb-1617-11cad6e316bf
# ╠═ea558d60-1642-11eb-0d64-9ff0b0223125
# ╠═13bf4660-15f7-11eb-0bac-676621168238
# ╠═145bea00-1643-11eb-2a19-fbfb2f2e6072
# ╠═4464d1d0-1643-11eb-1f61-a51dfd39197e
# ╟─7e2f48a0-1643-11eb-2e90-131b0e9580f5
# ╠═baeb2820-1645-11eb-3e15-0fe5abc28118
# ╠═c8088c4e-1645-11eb-164f-e5e814cbb6ff
# ╠═0eb90b00-1643-11eb-28b9-9f6f753d8861
# ╠═4f0e04e0-1647-11eb-0d85-fd2fc760b466
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
