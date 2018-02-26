import tensorflow as tf

def legendre(n, x):
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