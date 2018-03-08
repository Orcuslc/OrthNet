import tensorflow as tf
import torch
from _enum_dim import enum_dim as enum_dim


class Poly1d:
	"""
	1-dimensional orthogonal polynomials by three-term recursion.
	"""
	def __init__(self, module, degree, x, initial, recurrence):
		"""
		- input:
			- module: 'tensorflow' or 'pytorch' (case insensitive)
			- degree: degree of polynomial
			- x: a tensor of shape (n*1), where `n` is the number of sample points
			- initial: initial value, a list of two functions
			- recurrence: the function of recurrence, with three variables:
				P_{n+1} = f(P_{n}, P_{n-1}, n, x) 
			(initial and recurrence are functions on Tensors)
		"""
		self.module = module.lower()
		assert self.module in ['tensorflow', 'pytorch'], "Module should be either 'tensorflow' or 'pytorch'."
		assert degree >= 0 and isinstance(degree, int), "Degree should be a non-negative integer."
		assert len(initial) == 2, "Need two initial functions."
		if self.module == 'tensorflow':
			assert isinstance(x, tf.Variable) or isinstance(x, tf.Tensor), "x should be an isinstance of tensorflow.Variable or tensorflow.Tensor."
		else:
			assert isinstance(x, torch.autograd.Variable) or isinstance(x, torch.Tensor), "x should be an isinstance of torch.autograd.Variable or torch.Tensor."
		self.degree = degree
		self.x = x
		self.recurrence = recurrence
		self._list = [initial[0](self.x), initial[1](self.x)]

	def _compute(self, end):
		"""
		Compute polynomials up to order `end`
		"""
		degree = len(self._list)-1
		if end > degree:
			for i in range(degree, end):
				self._list.append(self.recurrence(self._list[-1], self._list[-2], i, self.x))

	@property
	def list(self):
		"""
		return a list of polynomials
		"""
		self._compute(self.degree)
		return self._list

	@property
	def tensor(self):
		"""
		return a tensor of polynomials
		"""
		if self.module == 'tensorflow':
			return tf.concat(self.list, axis = 1)
		else:
			return torch.cat(self.list, dim = 1)

	def update(self, newdegree):
		"""
		update polynomials to a higher degree
		"""
		self.degree = newdegree
		self._compute(newdegree)


class Poly:
	"""
	n-dimensional orthogonal polynomials by three-term recursion and tensor product
	"""
	def __init__(self, module, degree, x, initial, recurrence):
		"""
		- input:
			- module: 'tensorflow' or 'pytorch' (case insensitive)
			- degree: degree of polynomial
			- x: a tensor of shape (num_sample*num_parameter)
			- initial: initial value, a list of two functions
			- recurrence: the function of recurrence, with three variables:
				P_{n+1} = f(P_{n}, P_{n-1}, n, x) 
			(initial and recurrence are functions on Tensors)
		"""
		self.module = module.lower()
		assert self.module in ['tensorflow', 'pytorch'], "Module should be either 'tensorflow' or 'pytorch'."
		assert degree >= 0 and isinstance(degree, int), "Degree should be a non-negative integer."
		assert len(initial) == 2, "Need two initial functions."
		if self.module == 'tensorflow':
			assert isinstance(x, tf.Variable) or isinstance(x, tf.Tensor), "x should be an isinstance of tensorflow.Variable or tensorflow.Tensor."
		else:
			assert isinstance(x, torch.autograd.Variable) or isinstance(x, torch.Tensor), "x should be an isinstance of torch.autograd.Variable or torch.Tensor."
		self.degree = degree
		self.x = x
		self.recurrence = recurrence
		if self.module == 'tensorflow':
			self.dim = self.x.get_shape()[1].value
		else:
			self.dim = x.size()[1]
		self._poly_list = [Poly1d(module, degree, x[:, i], initial, recurrence).list for i in range(self.dim)]
		self._poly_list = []
		self._combination = None

	@property
	def combination(self):
		"""
		return the combination of 
		"""
		if not self._combination:
			self._combination = enum_dim(self.degree, self.dim)
		return self._combination

	def _compute_one_degree(self, degree):
		pass

	def _compute(self, end):
		pass

	@property
	def list(self):
		pass
	