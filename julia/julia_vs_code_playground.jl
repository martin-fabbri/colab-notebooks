function mean(x)
	acc = 0.0
	for i = 1:size(x)
		acc += x[i]
	end
	return acc / size(x)
end

mean([1, 2, 3])