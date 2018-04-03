import tensorflow as tf
import torch
from ._enum_dim import enum_dim as enum_dim


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

	def _compute(self, start, end):
		"""
		Compute polynomials from degree `start`(included) to `end`(included)
		"""
		for i in range(start-1, end):
			self._list.append(self.recurrence(self._list[-1], self._list[-2], i, self.x))

	@property
	def list(self):
		"""
		return a list of polynomials
		"""
		self._compute(len(self._list), self.degree)
		return [self._list[0]] if self.degree == 0 else self._list

	@property
	def tensor(self):
		"""
		return a tensor of polynomials
		"""
		if self.module == 'tensorflow':
			return tf.transpose(tf.concat(self.list, axis = 1))
		else:
			return torch.transpose(torch.cat(self.list, dim = 1))

	def update(self, newdegree):
		"""
		update polynomials to a higher degree
		"""
		self._compute(self.degree+1, newdegree)
		self.degree = newdegree

	def quadrature(self, func, weight):
		if self.module == 'tensorflow':
			weight = tf.reshape(tf.constant(weight), shape = [1, -1])
			return tf.matmul(weight, tf.multiply(func(self.x), self.tensor))
		else:
			weight = torch.tensor(weight, shape = [1, -1])
			return torch.matmul(weight, func(self.x)*self.tensor)



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
		self.initial = initial
		self.recurrence = recurrence
		if self.module == 'tensorflow':
			self.dim = self.x.get_shape()[1].value
		else:
			self.dim = x.size()[1]
		self._init()
		self._list = []

	def _init(self):
		self._comb()
		self._poly1d = [Poly1d(self.module, self.degree, self.x[:, i], self.initial, self.recurrence) for i in range(self.dim)]

	def _comb(self):
		comb = enum_dim(self.degree, self.dim)
		self._index = comb[0]
		self._combination = comb[1:]

	def _compute(self, start, end):
		"""
		compute polynomials from degree `start`(included) and `end`(included).
		"""
		if end == self.degree:
			comb = self._combination[self._index[start]:]
		else:
			comb = self._combination[self._index[start]:self._index[end]+1]
		res = []
		for c in comb:
			poly = tf.constant(1, dtype = tf.float64)
			if self.module == 'tensorflow':
				for i in range(len(c)):	
					poly = tf.multiply(poly, self._poly1d[i].list[c[i]])
				poly = tf.reshape(poly, [-1, 1])
			else:
				for i in range(len(c)):
					poly = poly*self._poly1d[i].list[c[i]]
				poly.unsqueeze_(1)
			res.append(poly)
		return res

	@property
	def length(self):
		"""
		return the number of polynomials
		"""
		return len(self.combination)

	@property
	def list(self):
		"""
		return a list of polynomials
		"""
		if not self._list:
			self._list.extend(self._compute(0, self.degree))
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
		update to a higher degree
		"""
		if newdegree > self.degree:
			original_degree = self.degree
			self.degree = newdegree
			self._comb()
			for i in range(self.dim):
				self._poly1d[i].update(newdegree)
			self._list.extend(self._compute(original_degree+1, newdegree))

	@property
	def index(self):
		"""
		return the index of the first combination of each degree
		"""
		return self._index

	@property
	def combination(self):
		"""
		return all combinations of all degrees
		"""
		return self._combination

	def get_one_degree_combination(self, degree):
		"""
		return all combination of a given degree
		"""
		assert degree <= self.degree, "Degree should be less or equal than the highest degree."
		if degree == self.degree:
			return self._combination[self._index[degree]:]
		else:
			return self._combination[self._index[degree]:self._index[degree+1]]

	def get_combination(self, start, end):
		"""
		return all combination of degrees from `start`(included)  to `end`(included)
		"""
		res = []
		for degree in range(start, end+1):
			res.extend(self.get_one_degree_combination(degree))
		return tuple(res)

	def get_poly(self, start, end):
		"""
		get polynomials from degree `start`(included) to `end`(included)
		"""
		assert start >= 0 and end <= self.degree, "Degree should be less or equal than highest degree"
		if end == self.degree:
			return self.tensor[:, self._index[start]:]
		else:
			return self.tensor[:, self._index[start]:self._index[end+1]]

	def eval(self, coeff):
		"""
		return the values with coefficients given
		"""
		assert len(coeff) == self.length, "Coefficients should have a length equal to number of polynomials"
		if self.module == 'tensorflow':
			coeff = tf.reshape(tf.constant(coeff), shape = [-1, 1])
			return tf.matmul(self.tensor, coeff)
		else:
			coeff = torch.Tensor(coeff, shape = [-1, 1])
			return torch.matmul(self.tensor, coeff)

	def quadrature(self, func, weight):
		if self.module == 'tensorflow':
			weight = tf.reshape(tf.constant(weight), shape = [1, -1])
			return tf.matmul(weight, tf.multiply(func(self.x), self.tensor))
		else:
			weight = torch.tensor(weight, shape = [1, -1])
			return torch.matmul(weight, func(self.x)*self.tensor)