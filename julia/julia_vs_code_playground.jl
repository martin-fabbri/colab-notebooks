function myclamp(x)
	return x < 0 ? 0 : x > 1 ? 1 : x 
end

myclamp(0.33)