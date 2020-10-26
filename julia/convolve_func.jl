using Statistics
using Images
using FFTW
using Plots
using DSP
using ImageFiltering
using OffsetArrays

function clamp_at_boundary(M, i, j)
	return M[
		clamp(i, 1, size(M, 1)),
		clamp(j, 1, size(M, 2)),
	]
end

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
						kernel[k, l] * M_index_function(M, i - k, j - l)
						for k in -half_height:-half_height + height - 1
						for l in -half_width:-half_width + width - 1
					])
		end
	end
	new_image
end

K = OffsetArray(gaussian((3,3), 0.25), -1:1, -1:1)
U = rand(1.0:100.0, 6, 6);
convolve(U, K)
