from __future__ import print_function
from ..backend import TensorflowBackend, TorchBackend, NumpyBackend

from ..utils import enum_dim, timeit


def poly1d(x, degree, initial, recurrence):
	"""
	Generate 1d orthogonal dimensional polynomials via three-term recurrence

	Input:
		- x: argument tensor
		- degree: highest degree
		- initial: initial 2 polynomials f_0, f_1
		- recurrence: the recurrence relation, 
			x_{n+1} = recurrence(x_{n}, x_{n-1}, n, x)

	Return:
		a list of polynomials from order 0 to order `degree`
	"""
	polys = [initial[0](x), initial[1](x)]
	if degree == 0:
		return [polys[0]]
	for i in range(1, degree):
		polys.append(recurrence(polys[-1], polys[-2], i, x))
	return polys


class Poly(object):
	def __init__(self, backend, x, degree, initial, recurrence, combinations = None):
		self._x = x
		self._degree = degree
		self._initial = initial
		self._recurrence = recurrence
		self._backend = backend
		self._dim = self._backend.get_dims(self._x)[1]

		if combinations is not None:
			self._index_combinations = combinations
		else:
			self._index_combinations = enum_dim(self._degree, self._dim)
		self._index = self._index_combinations[0]
		self._combinations = self._index_combinations[1:]

		self._poly1d = [poly1d(x[:, i], self._degree, self._initial, self._recurrence) for i in range(self._dim)]
		self._list = []
		
	def _compute(self, start, end):
		polynomials = []
		for comb in self._combinations[self._index[start]:(None if end == self._degree else self._index[end]+1)]:
			poly = self._backend.ones_like(self._x[:, 0])
			for i in range(len(comb)):
				poly = self._backend.multiply(poly, self._poly1d[i][comb[i]])
			poly = self._backend.expand_dims(poly, axis = 1)
			polynomials.append(poly)
		return polynomials

	@property
	def list(self):
		if not self._list:
			self._list = self._compute(0, self._degree)
		return self._list

	@property
	def tensor(self):
		return self._backend.concatenate(self.list, axis = 1)


	def update(self, new_degree):
		if new_degree > self._degree:
			original_degree = self._degree
			self._degree = new_degree
			self._combinations = enum_dim(self._degree, self._dim)
			self._index = self._combinations[0]
			self._combinations = self._combinations[1:]
			self._poly1d = [poly1d(x[:, i], self._degree, self._initial, self._recurrence) for i in range(self._dim)]
			self._list.extend(self._compute(original_degree+1, self._degree))

	@property
	def combinations(self):
		return self._combinations

	@property
	def index(self):
		return self._index

	@property
	def length(self):
		return len(self._combinations)

	def _get_poly(self, start, end):
		"""
		get polynomials from degree `start`(included) to `end`(included)
		"""
		assert start >= 0 and end <= self._degree, "Degree should be less or equal than highest degree"
		return self.tensor[:, self._index[start]:(None if end == self._degree else self._index[end+1])]

	def tensor_of_degree(self, degree):
		"""
		degree: a number or iterator of target degrees
		"""
		if isinstance(degree, int):
			degree = [degree]
		return self._get_poly(degree[0], degree[-1])

	def eval(self, coefficients):
		shape = self._backend.get_dims(coefficients)
		if len(shape) == 1:
			coefficients = self._backend.reshape((shape[0], 1))
		return self._backend.matmul(self.tensor, coefficients)

	def quadrature(self, function, weight):
		shape = self._backend.get_dims(weight)
		if len(shape) == 1:
			weight = self._backend.reshape((shape[0], 1))
		return self._backend.matmul(weight, self._backend.multiply(function(self._x), self.tensor))