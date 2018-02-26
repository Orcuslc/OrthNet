def enumerate_dim(n, dim):
	"""
	enumerate dims;

	input:
		n: total order
		dim: dimension (number of variables)

	output:
		a 2D-list, each element a combination (a list)
	"""
	def dfs(res, cur, n, dim):
		if dim == 1:
			res.append(cur+[n])
			return
		for i in range(n+1):
			dfs(res, cur+[i], n-i, dim-1)
	res = []
	dfs(res, [], n, dim)
	return res