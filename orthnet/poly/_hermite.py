from ..utils import check_backend
from .polynomial import Poly

class Hermite(Poly):
	"""
	Hermite polynomials of the first kind (in probability theory)
	"""
	def __init__(self, x, degree, backend = None, **kw):
		"""
		input:
			- x: a tensor
			- degree: highest degree of polynomial
		"""
		if backend is None:
			self._backend = check_backend(x)
		else:
			self._backend = backend
		initial = [lambda x: self._backend.ones_like(x), lambda x: x]
		recurrence = lambda p1, p2, n, x: self._backend.multiply(x, p1) - p2*n
		Poly.__init__(self, self._backend, x, degree, initial, recurrence, **kw)


class Hermite2(Poly):
	"""
	Hermite polynomials of the second kind (in Physics)
	"""
	def __init__(self, x, degree, backend = None, **kw):
		"""
		input:
			- x: a tensor
			- degree: highest degree of polynomial
		"""
		if backend is None:
			self._backend = check_backend(x)
		else:
			self._backend = backend
		initial = [lambda x: self._backend.ones_like(x), lambda x: x*2]
		recurrence = lambda p1, p2, n, x: (self._backend.multiply(x, p1) - p2*n)*2
		Poly.__init__(self, self._backend, x, degree, initial, recurrence, **kw)