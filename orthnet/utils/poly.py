import tensorflow as tf
import torch
from multi_dim import enumerate_dim as enumerate_dim
from _enum_dim import enum_dim as enum_dim_cpp


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
		assert n >= 0 and isinstance(n, int), "Degree should be a non-negative integer."
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
		assert n >= 0 and isinstance(n, int), "Degree should be a non-negative integer."
		assert len(initial) == 2, "Need two initial functions."
		if self.module == 'tensorflow':
			assert isinstance(x, tf.Variable) or isinstance(x, tf.Tensor), "x should be an isinstance of tensorflow.Variable or tensorflow.Tensor."
		else:
			assert isinstance(x, torch.autograd.Variable) or isinstance(x, torch.Tensor), "x should be an isinstance of torch.autograd.Variable or torch.Tensor."
		self.degree = degree
		self.x = x
		self.recurrence = recurrence
		if self.module == 'tensorflow':
			self.shape = [i.value for i in self.x.get_shape()]
		else:
			self.shape = list(self.x.size())
		self._list = [[initial[0](x[:, i]), initial[1](x[:, i])] for i in range(self.shape[1])]

		