import tensorflow as tf
import torch
from multi_dim import enumerate_dim as enum_dim_py
from _enum_dim import enum_dim as enum_dim_cpp


class Poly1d:
	"""
	Base class, 1-dimensional orthogonal polynomials by three-term recursion.
	"""
	def __init__(self, module, n, x, initial, recurrence):
		"""
		- input:
			- module: 'tensorflow' or 'pytorch' (case insensitive)
			- n: order of highest polynomial
			- x: tensor for evaluating function values (should have dimension >= 2)
			- initial: initial value, a list of two functions
			- recurrence: the function of recurrence, with three variables:
				P_{n+1} = f(P_{n}, P_{n-1}, n, x) 
			(initial and recurrence are functions on Tensors)
		"""
		self.module = module.lower()
		assert self.module in ['tensorflow', 'pytorch'], "Module should be either 'tensorflow' or 'pytorch'."
		assert n >= 0 and isinstance(n, int), "Order should be a non-negative integer."
		assert len(initial) == 2, "Need two initial functions."
		if self.module == 'tensorflow':
			assert isinstance(x, tf.Variable) or isinstance(x, tf.Tensor), "x should be an isinstance of tensorflow.Variable or tensorflow.Tensor."
		else:
			assert isinstance(x, torch.autograd.Variable) or isinstance(x, torch.Tensor), "x should be an isinstance of torch.autograd.Variable or torch.Tensor."
		self.n = n
		self.x = x
		self.initial = initial
		self.recurrence = recurrence

	@property
	def polys_list(self):
		"""
		generate a list of function values in lexicographical order.

		output:
			[[p0(x1), p0(x2), .., p0(xm)], [p1(x1), p1(x2), .., p1(xm)], .., [pn(x1), pn(x2), .., pn(xm)]]
		"""
		if self.n == 0:
			return [initial[0](self.x)]
		elif self.n == 1:
			return [initial[0](x), initial[1](x)]
		else: 
			polys = [initial[0](x), initial[1](x)]
			for i in range(1, n):
				polys.append(self.recurrence(polys[-1], polys[-2], i, self.x))
			return polys

	@property
	def poly_tensor(self):
		"""
		generate a tensor of function values in lexicographical order.

		output:
			Tensor([[p0(x)], [p1(x)], .., [pn(x)]])
		"""
		if self.module == 'tensorflow':
			return tf.concat(self.polys_list, axis = 1)
		else:
			return torch.cat(self.polys_list, dim = 1)


class Poly:
	"""
	Base class, multi-dimensional orthogonal polynomials by three-term recursion and tensor product.
	"""
	def __init__(self, module, n, x, initial, recurrence):
		"""
		input:
			- module: 'tensorflow' or 'pytorch'
			- n: order of target polynomial
			- x: a list of tensors (as variables)
			- initial: initial value, a list of two functions
			- recurrence: the function of recurrence, with three variables:
				P_{n+1} = f(P_{n}, P_{n-1}, n, x) 
			(initial and recurrence are functions on Tensors)

		output:
			- y: a list of function values
		"""
		self.module = module.lower()
		assert self.module in ['tensorflow', 'pytorch'], "Module should be either 'tensorflow' or 'pytorch'."
		assert n >= 0 and isinstance(n, int), "Order should be a non-negative integer."
		assert len(initial) == 2, "Need two initial functions."
		assert isinstance(x, list) or isinstance(x, tuple), "x should be a list or a tuple of tensors."
		self.n = n
		self.x = x
		self.initial = initial
		self.recurrence = recurrence

	@property
	def polys_list(self):
		"""
		generate a list of half tensor product of function values in lexicographical order.

		example:
			>>> x = Poly(_, 2, [x, y], _)
			>>> x.polys_list
			>>> [[p0(x)p0(y)], [p1(x)p0(y)], [p0(x)p1(y)], [p2(x)p0(y)], [p1(x)p1(y)], [p0(x)p2(y)]]
		"""
		one_dim_polys, polys = [], []
		