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
end

# ╔═╡ 41d37ef0-164b-11eb-0940-636023edbe0d
md"""
## A concrete first taste of abstraction
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
	for i in 1:size(M, 1)
		for j in 1:size(M, 2)
			# (k, l) loop over the neighbouring pixels
			new_image[i, j] = sum([
						@inbounds kernel[k, l]
						for k in -half_height:-half_height + height - 1
						for l in -half_width:-half_width + width - 1
					])
		end
	end
	new_image
	
end
	
	

# ╔═╡ eb820660-16b9-11eb-11e4-21b42906ae4b


# ╔═╡ 9650d1b0-16b6-11eb-061b-d94361398d7b
begin
	K = gaussian((3,3), 0.25)
	U = rand(1:100, 6, 6);
	convolve(U, K)
end

# ╔═╡ 464ef8b0-16be-11eb-0ccd-7b2d18d77a0c
K

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
# ╠═eb820660-16b9-11eb-11e4-21b42906ae4b
# ╠═9650d1b0-16b6-11eb-061b-d94361398d7b
# ╠═464ef8b0-16be-11eb-0ccd-7b2d18d77a0c
