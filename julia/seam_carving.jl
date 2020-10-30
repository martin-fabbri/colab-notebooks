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

# ╔═╡ d2bf60aa-1a49-11eb-3706-5f7507fc511c
begin
	using Images
	using PlutoUI
	using ImageFiltering
	using Statistics
	using LinearAlgebra
end

# ╔═╡ b2a8e540-19af-11eb-0214-9d5f04ac2486
md"""
# Seam Carving

1. We use convolution with Sobel filters for "edge detection".
2. We use that to write an algorithm that removes "uninteresting" bits of an image in order to shrink it.
"""

# ╔═╡ 455ad8f4-1a4f-11eb-38aa-b9da92a25c17
md"""
## Utility functions
"""

# ╔═╡ 988f2d30-1a50-11eb-0a3d-a15827ed7248
function hbox(x, y, gap=16; sy=size(y), sx=size(x))
	w,h = (max(sx[1], sy[1]),
		   gap + sx[2] + sy[2])
	
	slate = fill(RGB(1,1,1), w,h)
	slate[1:size(x,1), 1:size(x,2)] .= RGB.(x)
	slate[1:size(y,1), size(x,2) + gap .+ (1:size(y,2))] .= RGB.(y)
	slate
end

# ╔═╡ 14fc8df0-1a64-11eb-13af-cdd3cfc72f2e
vbox(x,y, gap=16) = hbox(x', y')'

# ╔═╡ d5fa6566-1a4f-11eb-0528-a35576836b80
function shrink_image(image, ratio=5)
	height, width = size(image)
	new_height = height ÷ ratio - 1
	new_width = width ÷ ratio - 1
	list = [
		mean(image[
				ratio * i:ratio * (i + 1),
				ratio * j:ratio * (j + 1)
		])
		for i in 1:new_height, j in 1:new_width
	]
	#reshape(list, new_height, new_width)
end

# ╔═╡ ba4af658-1a4f-11eb-217a-7f9165d30dbd
function convolve(M, kernel)
	height, width = size(kernel)

	half_height = height ÷ 2
	half_width = width ÷ 2

	new_image = similar(M)

	# (i, j) loop over the original image
	m, n = size(M)
	@inbounds for i in 1:m
		for j in 1:n
			# (k, l) loop over the neighbouring pixels
			accumulator = 0 * M[1, 1]
			for k in -half_height:-half_height + height - 1
				for l in -half_width:-half_width + width - 1
					Mi = i - k
					Mj = j - l
					# First index into M
					if Mi < 1
						Mi = 1
					elseif Mi > m
						Mi = m
					end
					# Second index into M
					if Mj < 1
						Mj = 1
					elseif Mj > n
						Mj = n
					end

					accumulator += kernel[k, l] * M[Mi, Mj]
				end
			end
			new_image[i, j] = accumulator
		end
	end

	return new_image
end

# ╔═╡ 571690f6-1a4f-11eb-2cb7-8d75d2dfc5eb
function show_colored_array(array)
	pos_color = RGB(0.36, 0.82, 0.8)
	neg_color = RGB(0.99, 0.18, 0.13)
	to_rgb(x) = max(x, 0) * pos_color + max(-x, 0) * neg_color
	to_rgb.(array) / maximum(abs.(array))
end


# ╔═╡ 53bf61d0-19b4-11eb-2212-d91065dae28f
@bind image_url Select([
"https://cdn.shortpixel.ai/spai/w_1086+q_lossy+ret_img+to_webp/https://wisetoast.com/wp-content/uploads/2015/10/The-Persistence-of-Memory-salvador-deli-painting.jpg",
"https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Gustave_Caillebotte_-_Paris_Street%3B_Rainy_Day_-_Google_Art_Project.jpg/1014px-Gustave_Caillebotte_-_Paris_Street%3B_Rainy_Day_-_Google_Art_Project.jpg",		
"https://web.mit.edu/facilities/photos/construction/Projects/stata/1_large.jpg",
])

# ╔═╡ 53d99a46-19b4-11eb-21fa-859486e6289b
img = load(download(image_url))

# ╔═╡ 53f647ea-19b4-11eb-2356-755781371a3a
# arbitrarily choose the brightness of a pixel as a mean of rgb
brightness(c::AbstractRGB) = 0.5 * c.r + 0.59 * c.g + 0.11 * c.b

# ╔═╡ 65932dac-1a67-11eb-2b1d-a74cd795c111
function edgeness(img)
	Sy, Sx = Kernel.sobel()
	b = brightness.(img)
	∇y = convolve(b, Sy)
	∇x = convolve(b, Sx)
	sqrt.(∇x.^2 + ∇y.^2)
end

# ╔═╡ 5413928c-19b4-11eb-1890-230a8077f4ae
Gray.(img)

# ╔═╡ 542f1244-19b4-11eb-207c-131b5699ad24
Gray.(brightness.(img))

# ╔═╡ ec7a7c5e-1a52-11eb-0b3f-49a1c0c2ab4b
shrink_image(img)

# ╔═╡ 2b5e3d8c-1a64-11eb-3a64-cb4e1c314db2
show_colored_array(brightness.(img))

# ╔═╡ ad7c6f50-1a4b-11eb-35fe-a7331d599b45
md"""
## Edge detection filter

We use the Sobel edge detection filter.

```math
\begin{align}

G_x &= \begin{bmatrix}
1 & 0 & -1 \\
2 & 0 & -2 \\
1 & 0 & -1 \\
\end{bmatrix}*A\\
G_y &= \begin{bmatrix}
1 & 2 & 1 \\
0 & 0 & 0 \\
-1 & -2 & -1 \\
\end{bmatrix}*A
\end{align}
```

Where: $A$ is the target image. We can think of these as derivates in the $x$ and $y$ directions.

Then we combine them by finding the magnitude of the **gradient** (in the sense of the multivatiate calculus) be defining.

$$G_\text{total} = \sqrt{G_x^2 + G_y^2}.$$
"""

# ╔═╡ ad6289f0-1a4b-11eb-1c1e-b1158e269574
Sy, Sx = Kernel.sobel();

# ╔═╡ ad4684c6-1a4b-11eb-3c63-5f888a1d834b
[show_colored_array(Sx) show_colored_array(Sy)]

# ╔═╡ ad2c6d34-1a4b-11eb-3f16-7332941d79fa
begin
	img_brightness = brightness.(img)
	∇x = convolve(img_brightness, Sx)
	∇y = convolve(img_brightness, Sy)
	hbox(show_colored_array(∇x), show_colored_array(∇y))
end

# ╔═╡ ad0d9044-1a4b-11eb-1a23-b7160b7793ed
vbox(
	hbox(img[200:end, 50:250], img[200:end, 50:250]), 
	hbox(show_colored_array.((∇x[200:end, 50:250], ∇y[200:end, 50:250]))...)
)

# ╔═╡ ac812a78-1a4b-11eb-216d-91a51c6b4742
begin
	edged = edgeness(img)
	hbox(img, Gray.(edged) / maximum(abs.(edged)))
end

# ╔═╡ ac652cc4-1a4b-11eb-2fee-dd2887e80796
md"""
## Seam carving idea

The idea of seam carving is to find the path from the top of the image to the bottom of the imaga where the path minimized the edgness.

In other words, this path minimized the number of edges it crosses.


"""

# ╔═╡ ac48d830-1a4b-11eb-3296-4d12c4153138


# ╔═╡ ac28a5f6-1a4b-11eb-16e8-e18a949846cb


# ╔═╡ ac002a2c-1a4b-11eb-3c4e-cb541f081ff1


# ╔═╡ abe1468e-1a4b-11eb-00e2-a394e9ae1504


# ╔═╡ abbef57a-1a4b-11eb-2c48-b12aa928410f


# ╔═╡ Cell order:
# ╠═d2bf60aa-1a49-11eb-3706-5f7507fc511c
# ╟─b2a8e540-19af-11eb-0214-9d5f04ac2486
# ╟─455ad8f4-1a4f-11eb-38aa-b9da92a25c17
# ╟─988f2d30-1a50-11eb-0a3d-a15827ed7248
# ╟─14fc8df0-1a64-11eb-13af-cdd3cfc72f2e
# ╟─d5fa6566-1a4f-11eb-0528-a35576836b80
# ╟─ba4af658-1a4f-11eb-217a-7f9165d30dbd
# ╟─571690f6-1a4f-11eb-2cb7-8d75d2dfc5eb
# ╟─65932dac-1a67-11eb-2b1d-a74cd795c111
# ╟─53bf61d0-19b4-11eb-2212-d91065dae28f
# ╠═53d99a46-19b4-11eb-21fa-859486e6289b
# ╠═53f647ea-19b4-11eb-2356-755781371a3a
# ╠═5413928c-19b4-11eb-1890-230a8077f4ae
# ╠═542f1244-19b4-11eb-207c-131b5699ad24
# ╠═ec7a7c5e-1a52-11eb-0b3f-49a1c0c2ab4b
# ╠═2b5e3d8c-1a64-11eb-3a64-cb4e1c314db2
# ╟─ad7c6f50-1a4b-11eb-35fe-a7331d599b45
# ╠═ad6289f0-1a4b-11eb-1c1e-b1158e269574
# ╠═ad4684c6-1a4b-11eb-3c63-5f888a1d834b
# ╟─ad2c6d34-1a4b-11eb-3f16-7332941d79fa
# ╟─ad0d9044-1a4b-11eb-1a23-b7160b7793ed
# ╟─ac812a78-1a4b-11eb-216d-91a51c6b4742
# ╠═ac652cc4-1a4b-11eb-2fee-dd2887e80796
# ╠═ac48d830-1a4b-11eb-3296-4d12c4153138
# ╠═ac28a5f6-1a4b-11eb-16e8-e18a949846cb
# ╠═ac002a2c-1a4b-11eb-3c4e-cb541f081ff1
# ╠═abe1468e-1a4b-11eb-00e2-a394e9ae1504
# ╠═abbef57a-1a4b-11eb-2c48-b12aa928410f
