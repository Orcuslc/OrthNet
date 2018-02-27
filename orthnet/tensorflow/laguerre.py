import tensorflow as tf
from ..utils.multi_dim import enumerate_dim
from .poly import *


def laguerre_list(n, x):
	"""
	Generate 1-dimensional Laguerre polynomial by three-term recursion:
		L_{n+1} = 1/(n+1)*((2n+1-x)L_n-n*L_{n-1})

	input:
		n: order of target polynomial
		x: tensor for evaluating function values

	output:
		y: a list of function values
	"""
	initial = [lambda x: tf.ones_like(x), lambda x: 1-x]
	recurrence = lambda p1, p2, n, x: (tf.multiply(2*n+1-x, p1)-n*p2)/(n+1)
	return poly_list(n, x, initial, recurrence)


def laguerre_tensor(n, x):
	"""
	Generate 1-dimensional Laguerre polynomial by three-term recursion:
		L_{n+1} = 1/(n+1)*((2n+1-x)L_n-n*L_{n-1})

	input:
		n: order of target polynomial
		x: tensor for evaluating function values

	output:
		y: a tensor of function values
	"""
	initial = [lambda x: tf.ones_like(x), lambda x: 1-x]
	recurrence = lambda p1, p2, n, x: (tf.multiply(2*n+1-x, p1)-n*p2)/(n+1)
	return poly_tensor(n, x, initial, recurrence)


def multi_dim_laguerre_list(n, var):
	"""
	multi dimensional Laguerre polynomials

	input:
		n: order of target polynomial
		x: a list of tensors 

	output:
		y: a list of function values

	>>> multi_dim_laguerre_list(2, [x, y])
	>>> [p0(x)p0(y), p1(x)p0(y), p0(x)p1(y), p2(x)p0(y), p1(x)p1(y), p0(x)p2(y)]
	# Each pi(x) is a laguerre polynomail of order i, (a list)
	"""
	return multi_dim_poly_list(n, var, laguerre_list)


def multi_dim_laguerre_tensor(n, var):
	"""
	multi dimensional Laguerre polynomials

	input:
		n: order of target polynomial
		x: a list of tensors 

	output:
		y: a tensor of function values

	>>> multi_dim_laguerre_tensor(2, [x, y])
	>>> Tensor([p0(x)p0(y), p1(x)p0(y), p0(x)p1(y), p2(x)p0(y), p1(x)p1(y), p0(x)p2(y)])
	# Each pi(x) is a laguerre polynomail of order i, (a list)
	"""
	return multi_dim_poly_tensor(n, var, laguerre_list)