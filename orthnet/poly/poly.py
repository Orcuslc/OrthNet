import tensorflow as tf
import torch
import numpy as np
from ..utils import enum_dim, timeit
import pickle


class Poly1d:
	"""
	1-dimensional orthogonal polynomials by three-term recursion.
	"""
	def __init__(self, module, degree, x, initial, recurrence, dtype = 'float32'):
		"""
		- input:
			- module: 'tensorflow', 'torch', 'torch.cuda' or 'numpy' (case insensitive)
			- degree: degree of polynomial
			- x: a tensor of shape (n*1), where `n` is the number of sample points
			- initial: initial value, a list of two functions
			- recurrence: the function of recurrence, with three variables:
				P_{n+1} = f(P_{n}, P_{n-1}, n, x) 
			(initial and recurrence are functions on Tensors)
			- dtype: 'float32' or 'float64'
		"""
		self._module = module.lower()
		assert self._module in ['tensorflow', 'torch', 'torch.cuda', 'numpy'], "Module should be one of ['tensorflow', 'torch', 'torch.cuda', numpy']."
		assert degree >= 0 and isinstance(degree, int), "Degree should be a non-negative integer."
		assert len(initial) == 2, "Need two initial functions."
		assert dtype in ['float32', 'float64'], "dtype should be either 'float32' or 'float64'"

		self._degree = degree
		self._x = x
		self._recurrence = recurrence
		self._dtype = dtype
		self._list = [initial[0](self._x), initial[1](self._x)]

	def _compute(self, start, end):
		"""
		Compute polynomials from degree `start`(included) to `end`(included)
		"""
		for i in range(start-1, end):
			self._list.append(self._recurrence(self._list[-1], self._list[-2], i, self._x))

	@property
	def list(self):
		"""
		return a list of polynomials
		"""
		self._compute(len(self._list), self._degree)
		return [self._list[0]] if self._degree == 0 else self._list

	@property
	def tensor(self):
		"""
		return a tensor of polynomials
		"""
		if self._module == 'tensorflow':
			return tf.concat(self.list, axis = 1)
		elif self._module == 'torch':
			return torch.cat(self.list, dim = 1)
		else:
			return np.concatenate(self.list, axis = 1)

	def update(self, newdegree):
		"""
		update polynomials to a higher degree
		"""
		self._compute(self._degree+1, newdegree)
		self._degree = newdegree


class Poly:
	"""
	n-dimensional orthogonal polynomials by three-term recursion and tensor product
	"""
	def __init__(self, module, degree, x, initial, recurrence, dtype = 'float32', loglevel = 0, index_comb = None):
		"""
		- input:
			- module: ['tensorflow', 'torch', 'numpy']
			- degree: degree of polynomial
			- x: a tensor of shape (num_sample*num_parameter)
			- initial: initial value, a list of two functions
			- recurrence: the function of recurrence, with four variables:
				P_{n+1} = f(P_{n}, P_{n-1}, n, x) 
				(initial and recurrence are functions on Tensors)
			- dtype: 'float32' or 'float64'
			- loglevel: 0 or 1
			- index_comb: combination of tensor product indices. If index_comb == None, the class will generate a new combination.
		"""
		self._module = module.lower()
		assert self._module in ['tensorflow', 'torch', 'torch.cuda', 'numpy'], "Module should be one of ['tensorflow', 'torch', 'torch.cuda', 'numpy']."
		assert degree >= 0 and isinstance(degree, int), "Degree should be a non-negative integer."
		assert len(initial) == 2, "Need two initial functions."
		assert dtype in ['float32', 'float64'], "dtype should be either 'float32' or 'float64'"

		self._degree = degree
		self._x = x
		self._initial = initial
		self._recurrence = recurrence
		self._dtype = dtype
		self._loglevel = loglevel

		if self._module == 'tensorflow':
			self.dim = self._x.get_shape()[1].value
		elif self._module == 'torch':
			self.dim = x.size()[1]
		elif self._module == 'numpy':
			self.dim = x.shape[1]

		if index_comb == None:
			self._comb()
		else:
			self._index = index_comb[0]
			self._combination = index_comb[1:]

		self._poly1d = [Poly1d(self._module, self._degree, self._x[:, i], self._initial, self._recurrence, self._dtype).list for i in range(self.dim)]
		self._list = []	

	def _comb(self):
		@timeit(self._loglevel)
		def __comb(self):
			comb_file_name = "comb_"+str(self._degree)+"_"+str(self.dim)
			try:
				comb = pickle.load(open(comb_file_name, "rb"))
			except FileNotFoundError:
				comb = enum_dim(self._degree, self.dim)
				pickle.dump(comb, open(comb_file_name, "wb"))
			except EOFError:
				comb = enum_dim(self._degree, self.dim)
				pickle.dump(comb, open(comb_file_name, "wb"))
			self._index = comb[0]
			self._combination = comb[1:]
		__comb(self)

	def _compute(self, start, end):
		"""
		compute polynomials from degree `start`(included) and `end`(included).
		"""
		@timeit(self._loglevel)
		def __compute(self, start, end):
			if end == self._degree:
				comb = self._combination[self._index[start]:]
			else:
				comb = self._combination[self._index[start]:self._index[end]+1]
			res = []
			for c in comb:
				if self._module == 'tensorflow':
					if self._dtype == 'float32':
						poly = tf.constant(1, dtype = tf.float32)
					else: 
						poly = tf.constant(1, dtype = tf.float64)
					for i in range(len(c)):	
						poly = tf.multiply(poly, self._poly1d[i][c[i]])
					poly = tf.reshape(poly, [-1, 1])
				elif self._module == 'torch':
					poly = torch.ones_like(self._x[:, 0])
					for i in range(len(c)):
						poly = poly*self._poly1d[i][c[i]]
					poly.unsqueeze_(1)
				# elif self._module == 'torch.cuda':
				# 	num_x = self._x.size()[0]
				# 	one_x = np.ones(num_x)
				# 	if self._dtype == 'float32':
				# 		poly = torch.cuda.FloatTensor(one_x)
				# 	else:
				# 		poly = torch.cuda.DoubleTensor(one_x)
				# 	for i in range(len(c)):
				# 		poly = poly*self._poly1d[i][c[i]]
				# 	poly.unsqueeze_(1)
				elif self._module == 'numpy':
					if self._dtype == 'float32':
						poly = np.ones([self._x.shape[0]], dtype = np.float32)
					else:
						poly = np.ones([self._x.shape[0]], dtype = np.float64)
					for i in range(len(c)):
						poly = poly*self._poly1d[i][c[i]]
					poly = np.expand_dims(poly, axis = 1)
				res.append(poly)
			return res
		return __compute(self, start, end)

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
			self._list.extend(self._compute(0, self._degree))
		return self._list

	@property
	def tensor(self):
		"""
		return a tensor of polynomials
		"""
		if self._module == 'tensorflow':
			return tf.concat(self.list, axis = 1)
		elif self._module == 'torch': 
			return torch.cat(self.list, dim = 1)
		elif self._module == 'numpy':
			return np.concatenate(self.list, axis = 1)

	def update(self, newdegree):
		"""
		update to a higher degree
		"""
		if newdegree > self._degree:
			original_degree = self._degree
			self._degree = newdegree
			self._comb()
			for i in range(self.dim):
				self._poly1d[i].update(newdegree)
			self._list.extend(self._compute(original_degree+1, newdegree))

	@property
	def index_comb(self):
		"""
		return index and combination for future usage
		"""
		if not self._index or not self._combination:
			self._comb()
		return (self._index, self._combination)

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
		assert degree <= self._degree, "Degree should be less or equal than the highest degree."
		if degree == self._degree:
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
		assert start >= 0 and end <= self._degree, "Degree should be less or equal than highest degree"
		if end == self._degree:
			return self.tensor[:, self._index[start]:]
		else:
			return self.tensor[:, self._index[start]:self._index[end+1]]

	def eval(self, coeff):
		"""
		return the values with coefficients given
		"""
		assert len(coeff) == self.length, "Coefficients should have a length equal to number of polynomials"
		if self._module == 'tensorflow':
			coeff = tf.reshape(tf.constant(coeff), shape = [-1, 1])
			return tf.matmul(self.tensor, coeff)
		elif self._module == 'torch':
			if self._dtype == 'float32':
				coeff = torch.FloatTensor(coeff, shape = [-1, 1])
			else:
				coeff = torch.DoubleTensor(coeff, shape = [-1, 1])
			coeff = type(self._x)(coeff).view(-1, 1)
			return torch.matmul(self.tensor, coeff)
		# elif self._module == 'torch.cuda':
		# 	if self._dtype == 'float32':
		# 		coeff = torch.cuda.FloatTensor(coeff, shape = [-1, 1])
		# 	else:
		# 		coeff = torch.cuda.DoubleTensor(coeff, shape = [-1, 1])
		# 	return torch.matmul(self.tensor, coeff)
		elif self._module == 'numpy':
			coeff = coeff.reshape((-1, 1))
			return np.dot(self.tensor, coeff)

	def quadrature(self, func, weight):
		if self._module == 'tensorflow':
			weight = tf.reshape(tf.constant(weight), shape = [1, -1])
			return tf.matmul(weight, tf.multiply(func(self._x), self.tensor))
		elif self._module == 'torch':
			if self._dtype == 'float32':
				weight = torch.FloatTensor(weight, shape = [-1, 1])
			else:
				weight = torch.DoubleTensor(weight, shape = [-1, 1])
			weight = type(self._x)(weight).view(-1, 1)
			return torch.matmul(weight, func(self._x)*self.tensor)
		# elif self._module == 'torch.cuda':
		# 	if self._dtype == 'float32':
		# 		weight = torch.cuda.FloatTensor(weight, shape = [-1, 1])
		# 	else:
		# 		weight = torch.cuda.DoubleTensor(weight, shape = [-1, 1])
		# 	return torch.matmul(weight, func(self._x)*self.tensor)
		elif s:
			weight = weight.reshape((1, -1))
			return np.dot(weight, func(self._x)*self.tensor)
