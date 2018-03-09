def enumerate_dim(n, dim):
	"""
	enumerate dims:
		find DIM nonnegative numbers, the sum of which is n, return result in lexicographical order.

	input:
		n: total order
		dim: dimension (number of variables)

	output:
		a 2D-list, each element a combination (a list)

	>>> enumerate_dim(3, 2)
	>>> [[3, 0], [2, 1], [1, 2], [0, 3]]
	"""
	def dfs(res, cur, n, dim):
		if dim == 1:
			res.append(cur+[n])
			return
		for i in reversed(range(n+1)):
			dfs(res, cur+[i], n-i, dim-1)
	res = []
	dfs(res, [], n, dim)
	# print('Order:', res)
	return res

def enum_dim(n, dim):
	"""
	enumerate dims:
		find DIM nonnegative numbers, the sum of which <= n, return result in lexicographical order.

	input:
		n: total order
		dim: dimension (number of variables)

	output:
		a 2D-list, each element a combination (a list)

	>>> enum_dim(3, 2)
	>>> [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [3, 0], [2, 1], [1, 2], [0, 3]]
	"""
	res = [[[0 for i in range(dim)]]]
	for i in range(n):
		cur = []
		for comb in res[-1]:
			for j in range(len(comb)):
				tmp = comb[:]
				tmp[j] += 1
				flag = 1
				for k in cur:
					if tmp == k:
						flag = 0
						break
				if flag:
					cur.append(tmp)
		res.append(cur)
	return res


def dim(n, d):
	res = []
	for i in range(n+1):
		res.append(enum_dim(i, d))
	return res