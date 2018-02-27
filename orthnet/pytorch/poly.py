import torch
from ..utils.multi_dim import enumerate_dim


def poly_list(n, x, initial, recurrence):
	"""
	Base function to generate Orthogonal Polynomials of one dimension, using Three-Term Recursion.

	- input:
		- n: order of highest polynomial
		- x: tensor for evaluating function values
		- initial: initial value, a list of two functions
		- recurrence: the function of recurrence, with three variables:
			P_{n+1} = f(P_{n}, P_{n-1}, n, x) 
		(initial and recurrence are functions on torch.Tensor)

	- output:
		- a list of function values
	"""
	assert n >= 0 and isinstance(n, int), "Order should be a non-negative integer."
	assert isinstance(x, torch.autograd.Variable) or isinstance(x, torch.Tensor), "x should be an isinstance of torch.autograd.Variable or torch.Tensor."
	assert len(initial) == 2, "Need two initial functions."
	if n == 0:
		return [initial[0](x)]
	elif n == 1:
		return [initial[0](x), initial[1](x)]
	polys = [initial[0](x), initial[1](x)]
	for i in range(1, n):
		polys.append(recurrence(polys[-1], polys[-2], i, x))
	return polys


def poly_tensor(n, x, initial, recurrence):
	"""
	Base function to generate Orthogonal Polynomials of one dimension, using Three-Term Recursion.

	input:
		- n: order of highest polynomial
		- x: tensor for evaluating function values
		- initial: initial value, a list of two functions
		- recurrence: the function of recurrence, with three variables:
			P_{n+1} = f(P_{n}, P_{n-1}, n, x) 
		(initial and recurrence are functions on torch.Tensor)

	output:
		- a tensor of function values
	"""
	assert n >= 0 and isinstance(n, int), "Order should be a non-negative integer."
	assert isinstance(x, torch.autograd.Variable) or isinstance(x, torch.Tensor), "x should be an isinstance of torch.autograd.Variable or torch.Tensor."
	assert len(initial) == 2, "Need two initial functions."
	if n == 0:
		return initial[0](x)
	elif n == 1:
		return torch.cat(initial[0](x), initial[1](x), dim = 1)
	polys = [initial[0](x), initial[1](x)]
	for i in range(1, n):
		polys.append(recurrence(polys[-1], polys[-2], i, x))
	return torch.cat(polys, dim = 1)


def multi_dim_poly_list(n, var, poly):
	"""
	Base function to generate multi-dimensional Orthogonal Polynomials

	input:
		- n: order of target polynomial
		- x: a list of tensors (as variables)
		- poly: the target type of polynomial, a function to generate a `list` of polynomials

	output:
		- y: a list of function values

	>>> multi_dim_poly_list(2, [x, y], legendre_list)
	>>> [p0(x)p0(y), p1(x)p0(y), p0(x)p1(y), p2(x)p0(y), p1(x)p1(y), p0(x)p2(y)] 
	# Each pi(x) is a legendre polynomail of order i, (a list)
	# `legendre_list` is defined in `layer.py`
	"""
	one_dim_polys, polys = [], []
	for x in var:
		one_dim_polys.append(poly(n, x))
	for i in range(n+1):
		dim_combinations = enumerate_dim(i, len(var))
		for comb in dim_combinations:
			apoly = 1.
			for i in range(len(comb)):
				apoly = apoly*one_dim_polys[i][comb[i]]
			polys.append(apoly)
	return polys


def multi_dim_poly_tensor(n, var, poly):
	"""
	Base function to generate multi-dimensional Orthogonal Polynomials

	input:
		- n: order of target polynomial
		- x: a list of tensors (as variables)
		- poly: the target type of polynomial, a function to generate a `list` of polynomials

	output:
		- y: a tensor of function values

	>>> multi_dim_poly_tensor(2, [x, y], legendre_list)
	>>> Tensor([p0(x)p0(y), p1(x)p0(y), p0(x)p1(y), p2(x)p0(y), p1(x)p1(y), p0(x)p2(y)])
	# Each pi(x) is a legendre polynomail of order i, (a tensor)
	# `legendre_tensor` is defined in `layer.py`
	"""
	one_dim_polys, polys = [], []
	for x in var:
		one_dim_polys.append(poly(n, x))
	for i in range(n+1):
		dim_combinations = enumerate_dim(i, len(var))
		for comb in dim_combinations:
			apoly = 1.
			for i in range(len(comb)):
				apoly = apoly*one_dim_polys[i][comb[i]]
			polys.append(apoly)
	return torch.cat(polys, dim = 1)