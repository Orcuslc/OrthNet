from ..backend import NumpyBackend, TensorflowBackend, TorchBackend

from .polynomial import Poly

class Chebyshev(Poly):
	"""
	Chebyshev polynomials of the fist kind
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
		recurrence = lambda p1, p2, n, x: self._backend.multiply(x, p1)*2 - p2
		Poly.__init__(self, self._backend, x, degree, initial, recurrence, *args, **kw)

class Chebyshev2(Poly):
	"""
	Chebyshev polynomials of the second kind
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
		initial = [lambda x: self._backend.ones_like(x), lambda x: x*2]
		recurrence = lambda p1, p2, n, x: self._backend.multiply(x, p1)*2 - p2
		Poly.__init__(self, self._backend, x, degree, initial, recurrence, *args, **kw)