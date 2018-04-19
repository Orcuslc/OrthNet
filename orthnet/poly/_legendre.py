from ..backend import NumpyBackend, TensorflowBackend, TorchBackend

from .polynomial import Poly
from math import sqrt

class Legendre(Poly):
	"""
	Legendre Polynomials
	"""
	def __init__(self, x, degree, *args, **kw):
		"""
		input:
			- x: a tensor
			- degree: highest degree of polynomial
		"""
		self._all_backends = list(filter(lambda backend: backend.is_available(), [TensorflowBackend(), TorchBackend(), NumpyBackend()]))
		self._backend = None
		for backend in self._all_backends:
			if backend.is_compatible(x):
				self._backend = backend
				break
		if self._backend is None:
			raise TypeError("Cannot determine backend from input arguments of type `{1}`. Available backends are {2}".format(type(self.x), ", ".join([str(backend) for backend in self._all_backends])))
		initial = [lambda x: self._backend.ones_like(x), lambda x: x]
		recurrence = lambda p1, p2, n, x: (self._backend.multiply(x, p1)*(2*n+1)-p2*n)/(n+1)
		Poly.__init__(self, self._backend, x, degree, initial, recurrence, *args, **kw)


class Legendre_Normalized(Poly):
	"""
	Normalized Legendre Polynomials with inner product be 1 if n == m.
	"""
	def __init__(self, x, degree, *args, **kw):
		"""
		input:
			- x: a tensor
			- degree: highest degree of polynomial
		"""
		self._all_backends = list(filter(lambda backend: backend.is_available(), [TensorflowBackend(), TorchBackend(), NumpyBackend()]))
		self._backend = None
		for backend in self._all_backends:
			if backend.is_compatible(x):
				self._backend = backend
				break
		if self._backend is None:
			raise TypeError("Cannot determine backend from input arguments of type `{1}`. Available backends are {2}".format(type(self.x), ", ".join([str(backend) for backend in self._all_backends])))
		initial = [lambda x: self._backend.ones_like(x)*sqrt(1/2), lambda x: x*sqrt(3/2)]
		recurrence = lambda p1, p2, n, x: (self._backend.multiply(x, p1)*sqrt((2*n+1)*(2*n+3))-p2*n*sqrt((2*n+3)/(2*n-1)))/(n+1)
		Poly.__init__(self, self._backend, x, degree, initial, recurrence, *args, **kw)