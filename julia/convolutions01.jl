### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 3df9d1c0-164c-11eb-335f-edb4366d2625
begin
	using Statistics
	using Images
	using FFTW
	using Plots
	using DSP
	using ImageFiltering
	using PlutoUI
	using OffsetArrays
end

# ╔═╡ 41d37ef0-164b-11eb-0940-636023edbe0d
md"""
## Image Convolutions
"""

# ╔═╡ f8db5200-1680-11eb-1d10-cbc2b31fc024
function shrink_image(image, ratio=5)
	(height, width) = size(image)
	new_height = height ÷ ratio - 1
	new_width = width ÷ ratio - 1
	list = [
			mean(image[
				ratio * i:ratio * (i + 1),
				ratio * j:ratio * (j + 1),
			])
			for j in 1:new_width
			for i in 1:new_height
	]
	reshape(list, new_height, new_width)
end

# ╔═╡ 35aa24a0-1680-11eb-065b-7973a0dab607
begin
	url = "https://upload.wikimedia.org/wikipedia/en/thumb/0/03/TheOreoCat.jpeg/900px-TheOreoCat.jpeg"
	download(url, "cat_in_a_hat.jpg")
	large_image = load("cat_in_a_hat.jpg")
	image = shrink_image(large_image, 7)
end

# ╔═╡ 2a905470-1687-11eb-30c3-5f635c6c0aeb
kernel = Kernel.gaussian((1, 1))

# ╔═╡ c2515c00-1687-11eb-3cf2-1385aa2ce3d6
function show_colored_kernel(kernel)
	to_rgb(x) = RGB(max(-x, 0), max(x,0), 0)
	to_rgb.(kernel) / maximum(abs.(kernel))
end

# ╔═╡ 0c4eb462-1688-11eb-0f56-cf09d1f53f0a
show_colored_kernel(kernel)

# ╔═╡ 9c21a2b0-16af-11eb-3fae-6d8927359b72
function clamp_at_boundary(M, i, j)
	return M[
		clamp(i, 1, size(M, 1)),
		clamp(j, 1, size(M, 2)),
	]
end

# ╔═╡ 307c6610-16b6-11eb-159c-f18f672027df
begin
	I = [1 2 3; 8 4 9]
	size(I, 2)
end

# ╔═╡ 1592aef0-1688-11eb-2d74-35ce0cb1ddc3
function convolve(M, kernel, M_index_function=clamp_at_boundary)
	height = size(kernel, 1)
	width = size(kernel, 2)
	
	half_height = height ÷ 2
	half_width = width ÷ 2
	
	new_image = similar(M)
	
	# (i, j) loop over the original image
	@inbounds for i in 1:size(M, 1)
		for j in 1:size(M, 2)
			# (k, l) loop over the neighbouring pixels
			new_image[i, j] = sum([
						kernel[k, l] * M_index_function(M, i - k, j - l)
						for k in -half_height:-half_height + height - 1
						for l in -half_width:-half_width + width - 1
					])
		end
	end
	return new_image
end
	
	

# ╔═╡ 9650d1b0-16b6-11eb-061b-d94361398d7b
begin
	K = OffsetArray(gaussian((3,3), 0.25), -1:1, -1:1)
	U = rand(1.0:100.0, 6, 6);
	convolve(U, K)
end

# ╔═╡ 765a986a-173c-11eb-29e6-cb155437d4ea
convolve(image, Kernel.gaussian((3, 3)))

# ╔═╡ 7767a37e-173c-11eb-39a8-0d1bd7be9077
convolve(image, Kernel.gaussian((10, 10)))

# ╔═╡ 7785d7e0-173c-11eb-11b4-ebc2a9c3815a
sharpen_kernel = centered([
	-0.5 -1.0 -0.5
	-1.0  7.0 -1.0
	-0.5 -1.0 -0.5
])

# ╔═╡ cea7f490-1746-11eb-2a66-8f75edaec775
edge_detection_kernel_horizontal = Kernel.sobel()[1]

# ╔═╡ b9733322-1747-11eb-1da7-3d4b992d60a2
show_colored_kernel(edge_detection_kernel_horizontal)

# ╔═╡ ed2204f0-1747-11eb-1662-9d8199f940a1
edge_detection_kernel_vertical = Kernel.sobel()[2]

# ╔═╡ 0018a2bc-1748-11eb-090f-8189d8958e3b
show_colored_kernel(edge_detection_kernel_vertical)

# ╔═╡ 25644f9e-1748-11eb-1785-b199708755e1
sum(edge_detection_kernel_vertical)

# ╔═╡ 64f95e42-1748-11eb-1048-7f168caea4b4
edge_enhanced_vertical = 3 * Gray.(abs.(convolve(image, edge_detection_kernel_vertical)))

# ╔═╡ bb0d4492-1748-11eb-05c6-0973e38a9f28
edge_enhanced_horizontal = 3 * Gray.(abs.(convolve(image, edge_detection_kernel_horizontal)))

# ╔═╡ c1823bbe-1745-11eb-10ad-a56948b9eaab
[image convolve(image, sharpen_kernel) convolve(convolve(image, sharpen_kernel), edge_detection_kernel_horizontal) convolve(convolve(image, sharpen_kernel), edge_detection_kernel_vertical) edge_enhanced_vertical edge_enhanced_horizontal]

# ╔═╡ 16d61752-1746-11eb-305e-81831983418b
sum(sharpen_kernel)

# ╔═╡ 77d0acd4-173c-11eb-28ea-5d1566fbb940


# ╔═╡ ea02e700-1745-11eb-3562-1d1fa9d3e075


# ╔═╡ ea23db4a-1745-11eb-07c4-5ffeafc03183


# ╔═╡ ea460db4-1745-11eb-1af5-1d206f5b617b


# ╔═╡ ea635bb2-1745-11eb-2631-d56be9f6e88c


# ╔═╡ ea80a71c-1745-11eb-3a0d-773e478efcc6


# ╔═╡ ea9dec78-1745-11eb-2473-bbdce2c4c96a


# ╔═╡ 77eb908a-173c-11eb-1323-8192758a3bf8


# ╔═╡ 780650a0-173c-11eb-00db-f7c0ec17cf52


# ╔═╡ 78a80050-173c-11eb-2fe8-6b70bb6a2507


# ╔═╡ Cell order:
# ╟─41d37ef0-164b-11eb-0940-636023edbe0d
# ╠═3df9d1c0-164c-11eb-335f-edb4366d2625
# ╠═f8db5200-1680-11eb-1d10-cbc2b31fc024
# ╠═35aa24a0-1680-11eb-065b-7973a0dab607
# ╠═2a905470-1687-11eb-30c3-5f635c6c0aeb
# ╠═c2515c00-1687-11eb-3cf2-1385aa2ce3d6
# ╠═0c4eb462-1688-11eb-0f56-cf09d1f53f0a
# ╠═9c21a2b0-16af-11eb-3fae-6d8927359b72
# ╠═307c6610-16b6-11eb-159c-f18f672027df
# ╠═1592aef0-1688-11eb-2d74-35ce0cb1ddc3
# ╠═9650d1b0-16b6-11eb-061b-d94361398d7b
# ╠═765a986a-173c-11eb-29e6-cb155437d4ea
# ╠═7767a37e-173c-11eb-39a8-0d1bd7be9077
# ╠═7785d7e0-173c-11eb-11b4-ebc2a9c3815a
# ╠═cea7f490-1746-11eb-2a66-8f75edaec775
# ╠═b9733322-1747-11eb-1da7-3d4b992d60a2
# ╠═ed2204f0-1747-11eb-1662-9d8199f940a1
# ╠═0018a2bc-1748-11eb-090f-8189d8958e3b
# ╠═25644f9e-1748-11eb-1785-b199708755e1
# ╠═64f95e42-1748-11eb-1048-7f168caea4b4
# ╠═bb0d4492-1748-11eb-05c6-0973e38a9f28
# ╠═c1823bbe-1745-11eb-10ad-a56948b9eaab
# ╠═16d61752-1746-11eb-305e-81831983418b
# ╠═77d0acd4-173c-11eb-28ea-5d1566fbb940
# ╠═ea02e700-1745-11eb-3562-1d1fa9d3e075
# ╠═ea23db4a-1745-11eb-07c4-5ffeafc03183
# ╠═ea460db4-1745-11eb-1af5-1d206f5b617b
# ╠═ea635bb2-1745-11eb-2631-d56be9f6e88c
# ╠═ea80a71c-1745-11eb-3a0d-773e478efcc6
# ╠═ea9dec78-1745-11eb-2473-bbdce2c4c96a
# ╠═77eb908a-173c-11eb-1323-8192758a3bf8
# ╠═780650a0-173c-11eb-00db-f7c0ec17cf52
# ╠═78a80050-173c-11eb-2fe8-6b70bb6a2507
