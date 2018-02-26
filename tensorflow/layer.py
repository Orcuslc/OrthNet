import tensorflow as tf
from ..utils.multi_dim import enumerate_dim


def legendre_list(n, x):
	"""
	Generate 1-dimensional Legendre polynomial by three-term recursion:
		P_{n+1} = 1/(n+1)*((2n+1)xP_n - nP_{n-1})

	input:
		n: order of target polynomial
		x: tensor for evaluating function values

	output:
		y: a list of function values
	"""
	assert n >= 0, "Order should >= 0."
	assert isinstance(x, tf.Variable) or isinstance(x, tf.Tensor), "x should be a instance of tf.Variable or tf.Tensor."
	if n == 0:
		return [tf.ones_like(x)]
	elif n == 1:
		return [tf.ones_like(x), x]
	polys = [tf.ones_like(x), x]
	for i in range(1, n):
		polys.append(((2*i+1)*tf.multiply(x, polys[-1])-i*polys[-2])/(i+1))
	return polys


def legendre_tensor(n, x):
	"""
	Generate 1-dimensional Legendre polynomial by three-term recursion:
		P_{n+1} = 1/(n+1)*((2n+1)xP_n - nP_{n-1})

	input:
		n: order of target polynomial
		x: tensor for evaluating function values

	output:
		y: a tensor of function values
	"""
	assert n >= 0, "Order should >= 0."
	assert isinstance(x, tf.Variable) or isinstance(x, tf.Tensor), "x should be a instance of tf.Variable or tf.Tensor."
	if n == 0:
		return tf.ones_like(x)
	elif n == 1:
		return tf.concat([tf.ones_like(x), x], axis = 1)
	polys = [tf.ones_like(x), x]
	for i in range(1, n):
		polys.append(((2*i+1)*tf.multiply(x, polys[-1])-i*polys[-2])/(i+1))
	return tf.concat(polys, axis = 1)


def multi_dim_legendre_list(n, var):
	"""
	multi dimensional legendre polynomials

	input:
		n: order of target polynomial
		x: a list of tensors 

	output:
		y: a list of function values

	>>> multi_dim_legendre_list(3, [x, y])
	>>> [p3(x)p0(y), p2(x)p1(y), p1(x)p2(y), p0(x)p3(y)] # Each pi(x) is a legendre polynomail of order i, (a tensor)
	"""
	one_dim_polys, polys = [], []
	for x in var:
		one_dim_polys.append(legendre_list(n, x))
	dim_combinations = enumerate_dim(n, len(var))
	for comb in dim_combinations:
		apoly = 1.
		for i in range(len(comb)):
			apoly = tf.multiply(apoly, one_dim_polys[i][comb[i]])
		polys.append(apoly)
	return polys


def multi_dim_legendre_tensor(n, var):
	"""
	multi dimensional legendre polynomials

	input:
		n: order of target polynomial
		x: a list of tensors 

	output:
		y: a tensor of function values

	>>> multi_dim_legendre_list(3, [x, y])
	>>> Tensor([p3(x)p0(y), p2(x)p1(y), p1(x)p2(y), p0(x)p3(y)]) # Each pi(x) is a legendre polynomail of order i, (a tensor)
	"""
	one_dim_polys, polys = [], []
	for x in var:
		one_dim_polys.append(legendre_list(n, x))
	dim_combinations = enumerate_dim(n, len(var))
	for comb in dim_combinations:
		apoly = 1.
		for i in range(len(comb)):
			apoly = tf.multiply(apoly, one_dim_polys[i][comb[i]])
		polys.append(apoly)
	return tf.concat(polys, axis = 1)
