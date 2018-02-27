import tensorflow as tf
from ..utils.multi_dim import enumerate_dim
from .poly import *


def chebyshev_list(n, x):
	"""
	Generate 1-dimensional Chebyshev polynomial (of the first kind) by three-term recursion:
		T_{n+1} = 2xT_n - T_{n-1}

	input:
		n: order of target polynomial
		x: tensor for evaluating function values

	output:
		y: a list of function values
	"""
	initial = [lambda x: tf.ones_like(x), lambda x: x]
	recurrence = lambda p1, p2, n, x: 2*tf.multiply(x, p1)-p2
	return poly_list(n, x, initial, recurrence)


def chebyshev_tensor(n, x):
	"""
	Generate 1-dimensional Chebyshev polynomial (of the first kind) by three-term recursion:
		T_{n+1} = 2xT_n - T_{n-1}

	input:
		n: order of target polynomial
		x: tensor for evaluating function values

	output:
		y: a tensor of function values
	"""
	initial = [lambda x: tf.ones_like(x), lambda x: x]
	recurrence = lambda p1, p2, n, x: 2*tf.multiply(x, p1)-p2
	return poly_tensor(n, x, initial, recurrence)


def multi_dim_chebyshev_list(n, var):
	"""
	multi dimensional Chebyshev polynomials

	input:
		n: order of target polynomial
		x: a list of tensors 

	output:
		y: a list of function values

	>>> multi_dim_chebyshev_list(2, [x, y])
	>>> [p0(x)p0(y), p1(x)p0(y), p0(x)p1(y), p2(x)p0(y), p1(x)p1(y), p0(x)p2(y)]
	# Each pi(x) is a chebyshev polynomial of order i, (a list)
	"""
	return multi_dim_poly_list(n, var, chebyshev_list)


def multi_dim_chebyshev_tensor(n, var):
	"""
	multi dimensional Chebyshev polynomials

	input:
		n: order of target polynomial
		x: a list of tensors 

	output:
		y: a tensor of function values

	>>> multi_dim_chebyshev_tensor(2, [x, y])
	>>> Tensor([p0(x)p0(y), p1(x)p0(y), p0(x)p1(y), p2(x)p0(y), p1(x)p1(y), p0(x)p2(y)])
	# Each pi(x) is a chebyshev polynomial of order i, (a tensor)
	"""
	return multi_dim_poly_tensor(n, var, chebyshev_list)


def chebyshev2_list(n, x):
	"""
	Generate 1-dimensional Chebyshev polynomial (of the second kind) by three-term recursion:
		U_{n+1} = 2xU_n - U_{n-1}

	input:
		n: order of target polynomial
		x: tensor for evaluating function values

	output:
		y: a list of function values
	"""
	initial = [lambda x: tf.ones_like(x), lambda x: 2*x]
	recurrence = lambda p1, p2, n, x: 2*tf.multiply(x, p1)-p2
	return poly_list(n, x, initial, recurrence)


def chebyshev2_tensor(n, x):
	"""
	Generate 1-dimensional Chebyshev polynomial (of the second kind) by three-term recursion:
		U_{n+1} = 2xU_n - U_{n-1}

	input:
		n: order of target polynomial
		x: tensor for evaluating function values

	output:
		y: a tensor of function values
	"""
	initial = [lambda x: tf.ones_like(x), lambda x: 2*x]
	recurrence = lambda p1, p2, n, x: 2*tf.multiply(x, p1)-p2
	return poly_tensor(n, x, initial, recurrence)


def multi_dim_chebyshev2_list(n, var):
	"""
	multi dimensional Chebyshev polynomials (of the second kind)

	input:
		n: order of target polynomial
		x: a list of tensors 

	output:
		y: a list of function values

	>>> multi_dim_chebyshev2_list(2, [x, y])
	>>> [p0(x)p0(y), p1(x)p0(y), p0(x)p1(y), p2(x)p0(y), p1(x)p1(y), p0(x)p2(y)]
	# Each pi(x) is a chebyshev polynomial of order i, (a list)
	"""
	return multi_dim_poly_list(n, var, chebyshev2_list)


def multi_dim_chebyshev2_tensor(n, var):
	"""
	multi dimensional Chebyshev polynomials (of the second kind)

	input:
		n: order of target polynomial
		x: a list of tensors 

	output:
		y: a tensor of function values

	>>> multi_dim_chebyshev2_tensor(2, [x, y])
	>>> Tensor([p0(x)p0(y), p1(x)p0(y), p0(x)p1(y), p2(x)p0(y), p1(x)p1(y), p0(x)p2(y)])
	# Each pi(x) is a chebyshev polynomial of order i, (a tensor)
	"""
	return multi_dim_poly_tensor(n, var, chebyshev2_list)