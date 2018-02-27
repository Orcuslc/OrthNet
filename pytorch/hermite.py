import torch
from ..utils.multi_dim import enumerate_dim
from .poly import *


def hermite_list(n, x):
	"""
	Generate 1-dimensional Hermite polynomial (in probability theory) by three-term recursion:
		He_{n+1} = xHe_n - nHe_{n-1}

	input:
		n: order of target polynomial
		x: tensor for evaluating function values

	output:
		y: a list of function values
	"""
	initial = [lambda x: torch.ones_like(x), lambda x: x]
	recurrence = lambda p1, p2, n, x: x*p1 - n*p2
	return poly_list(n, x, initial, recurrence)


def hermite_tensor(n, x):
	"""
	Generate 1-dimensional Hermite polynomial (in probability theory) by three-term recursion:
		He_{n+1} = xHe_n - nHe_{n-1}

	input:
		n: order of target polynomial
		x: tensor for evaluating function values

	output:
		y: a tensor of function values
	"""
	initial = [lambda x: torch.ones_like(x), lambda x: x]
	recurrence = lambda p1, p2, n, x: x*p1 - n*p2
	return poly_tensor(n, x, initial, recurrence)


def multi_dim_hermite_list(n, var):
	"""
	multi dimensional Hermite polynomials

	input:
		n: order of target polynomial
		x: a list of tensors 

	output:
		y: a list of function values

	>>> multi_dim_hermite_list(2, [x, y])
	>>> [p0(x)p0(y), p1(x)p0(y), p0(x)p1(y), p2(x)p0(y), p1(x)p1(y), p0(x)p2(y)]
	# Each pi(x) is an Hermite polynomail of order i, (a list)
	"""
	return multi_dim_poly_list(n, var, hermite_list)


def multi_dim_hermite_tensor(n, var):
	"""
	multi dimensional hermite polynomials

	input:
		n: order of target polynomial
		x: a list of tensors 

	output:
		y: a tensor of function values

	>>> multi_dim_hermite_tensor(2, [x, y])
	>>> Tensor([p0(x)p0(y), p1(x)p0(y), p0(x)p1(y), p2(x)p0(y), p1(x)p1(y), p0(x)p2(y)])
	# Each pi(x) is an Hermite polynomail of order i, (a tensor)
	"""
	return multi_dim_poly_tensor(n, var, hermite_list)


def hermite2_list(n, x):
	"""
	Generate 1-dimensional Hermite polynomial (in physics) by three-term recursion:
		H_{n+1} = 2xH_n - 2nH_{n-1}

	input:
		n: order of target polynomial
		x: tensor for evaluating function values

	output:
		y: a list of function values
	"""
	initial = [lambda x: torch.ones_like(x), lambda x: x]
	recurrence = lambda p1, p2, n, x: 2*(x*p1 - n*p2)
	return poly_list(n, x, initial, recurrence)


def hermite2_tensor(n, x):
	"""
	Generate 1-dimensional Hermite polynomial (in physics) by three-term recursion:
		H_{n+1} = 2xH_n - 2nH_{n-1}

	input:
		n: order of target polynomial
		x: tensor for evaluating function values

	output:
		y: a list of function values
	"""
	initial = [lambda x: torch.ones_like(x), lambda x: x]
	recurrence = lambda p1, p2, n, x: 2*(x*p1 - n*p2)
	return poly_tensor(n, x, initial, recurrence)


def multi_dim_hermite2_list(n, var):
	"""
	multi dimensional Hermite polynomials

	input:
		n: order of target polynomial
		x: a list of tensors 

	output:
		y: a list of function values

	>>> multi_dim_hermite_list(2, [x, y])
	>>> [p0(x)p0(y), p1(x)p0(y), p0(x)p1(y), p2(x)p0(y), p1(x)p1(y), p0(x)p2(y)]
	# Each pi(x) is an Hermite polynomail of order i, (a list)
	"""
	return multi_dim_poly_list(n, var, hermite2_list)


def multi_dim_hermite2_tensor(n, var):
	"""
	multi dimensional hermite polynomials

	input:
		n: order of target polynomial
		x: a list of tensors 

	output:
		y: a tensor of function values

	>>> multi_dim_hermite_tensor(2, [x, y])
	>>> Tensor([p0(x)p0(y), p1(x)p0(y), p0(x)p1(y), p2(x)p0(y), p1(x)p1(y), p0(x)p2(y)])
	# Each pi(x) is an Hermite polynomail of order i, (a tensor)
	"""
	return multi_dim_poly_tensor(n, var, hermite2_list)
