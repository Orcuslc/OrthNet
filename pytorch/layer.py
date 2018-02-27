import torch
from ..utils.multi_dim import enumerate_dim
from .poly import *


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
	initial = [lambda x: torch.ones_like(x), lambda x: x]
	recurrence = lambda p1, p2, n, x: ((2*n+1)*x*p1-n*p2)/(n+1)
	return poly_list(n, x, initial, recurrence)


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
	initial = [lambda x: torch.ones_like(x), lambda x: x]
	recurrence = lambda p1, p2, n, x: ((2*n+1)*x*p1-n*p2)/(n+1)
	return poly_tensor(n, x, initial, recurrence)


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
	return multi_dim_poly_list(n, var, legendre_list)


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
	return multi_dim_poly_tensor(n, var, legendre_list)