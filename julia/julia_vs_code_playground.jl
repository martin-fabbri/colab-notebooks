using Images

function mean_colors(image)
	h, w = size(image)
	image_size = h * w
	sum_r, sum_g, sum_b = (0, 0, 0)
	for i=1:h
		for j=1:w
			sum_r, sum_g, sum_b = (sum_r + image[i, j].r, sum_g + image[i, j].g, sum_b + image[i, j].b)
		end
	end
	return (sum_r / image_size, sum_g / image_size, sum_b / image_size)
end

cat = load("cat_in_a_hat.jpg")
mean_colors(cat)