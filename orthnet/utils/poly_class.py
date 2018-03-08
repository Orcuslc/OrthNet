import tensorflow as tf
import torch
from multi_dim import enumerate_dim as enumerate_dim
from _enum_dim import enum_dim as enum_dim_cpp


class Poly1d:
	"""
	Base class, 1-dimensional orthogonal polynomials by three-term recursion.
	"""
	def __init__(self, module, degree, x, initial, recurrence):
		"""
		- input:
			- module: 'tensorflow' or 'pytorch' (case insensitive)
			- degree: degree of polynomial
			- x: tensor for evaluating function values (should have dimension >= 2)
			- initial: initial value, a list of two functions
			- recurrence: the function of recurrence, with three variables:
				P_{n+1} = f(P_{n}, P_{n-1}, n, x) 
			(initial and recurrence are functions on Tensors)
		"""
		self.module = module.lower()
		assert self.module in ['tensorflow', 'pytorch'], "Module should be either 'tensorflow' or 'pytorch'."
		assert n >= 0 and isinstance(n, int), "Degree should be a non-negative integer."
		assert len(initial) == 2, "Need two initial functions."
		if self.module == 'tensorflow':
			assert isinstance(x, tf.Variable) or isinstance(x, tf.Tensor), "x should be an isinstance of tensorflow.Variable or tensorflow.Tensor."
		else:
			assert isinstance(x, torch.autograd.Variable) or isinstance(x, torch.Tensor), "x should be an isinstance of torch.autograd.Variable or torch.Tensor."
		self.degree = degree
		self.x = x
		self.initial = initial
		self.recurrence = recurrence
		self.poly_list = [self.initial[0](self.x), self.initial[1](self.x)]

	def _compute(self, start, end):
		if end == 0:
			return [self.initial[0](self.x)]
		elif end == 1:
			return [self.initial[0](self.x), self.initial[1](self.x)]
		else:
			polys = []
			for i in range(start, end+1):

	def update(self, new_degree):
		self.
				

	@property
	def list(self):
		"""
		generate a list of function values in lexicographical order.

		output:
			[[p0(x1), p0(x2), .., p0(xm)], [p1(x1), p1(x2), .., p1(xm)], .., [pn(x1), pn(x2), .., pn(xm)]]
		"""
		if self.poly_list:
			return self.poly_list
		if self.order == 0:
			self.poly_list = [initial[0](self.x)]
		elif self.order == 1:
			self.poly_list = [initial[0](x), initial[1](x)]
		else: 
			self.poly_list = [initial[0](x), initial[1](x)]
			for i in range(1, n):
				self.poly_list.append(self.recurrence(self.poly_list[-1], self.poly_list[-2], i, self.x))
			return self.poly_list

	@property
	def tensor(self):
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
			- x: a tensor, each row is a sample point, and each column is a feature (or variable).
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
		self.n = n
		self.x = x
		self.initial = initial
		self.recurrence = recurrence
		self.dims = []
		if self.module == 'tensorflow':
			self.n_features

	@property
	def list(self):
		"""
		generate a list of half tensor product of function values in lexicographical order.

		example:
			>>> x = Poly(_, 2, [x, y], _)
			>>> x.polys_list
			>>> [[p0(x)p0(y)], [p1(x)p0(y)], [p0(x)p1(y)], [p2(x)p0(y)], [p1(x)p1(y)], [p0(x)p2(y)]]
		"""
		one_dim_polys, polys = [], []
		if 
		# for index in range()
		# 	tmp = Poly1d(self.module, self.n, var, self.initial, self.recurrence)
		# 	one_dim_polys.append(tmp.list)
		if not self.dims:
			for i in range(n+1):
				self.dims.append(enumerate_dim_cpp())
			



# class Poly(Poly1d):
# 	"""
# 	Base class, multi-dimensional orthogonal polynomials by three-term recursion and tensor product.
# 	"""
# 	def __init__(self, module, n, x, initial, recurrence):
# 		"""
# 		input:
# 			- module: 'tensorflow' or 'pytorch'
# 			- n: order of target polynomial
# 			- x: a list of tensors (as variables)
# 			- initial: initial value, a list of two functions
# 			- recurrence: the function of recurrence, with three variables:
# 				P_{n+1} = f(P_{n}, P_{n-1}, n, x) 
# 			(initial and recurrence are functions on Tensors)

# 		output:
# 			- y: a list of function values
# 		"""
# 		assert isinstance(x, list) or isinstance(x, tuple), "x should be a list or a tuple of tensors."
# 		super().__init__(self, module, n, x[0], initial, recurrence)
# 		self.x = x # Override super.x

# 	@property
# 	def list(self):
# 		"""
# 		generate a list of half tensor product of function values in lexicographical order.

# 		example:
# 			>>> x = Poly(_, 2, [x, y], _)
# 			>>> x.polys_list
# 			>>> [[p0(x)p0(y)], [p1(x)p0(y)], [p0(x)p1(y)], [p2(x)p0(y)], [p1(x)p1(y)], [p0(x)p2(y)]]
# 		"""
# 		one_dim_polys, polys = [], []
# 		for var in self.x:
# 			tmp = Poly1d(self.module, n, x, initial, recurrence)
# 			one_dim_polys.append()