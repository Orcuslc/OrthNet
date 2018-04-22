from ..utils import check_backend
from .polynomial import Poly

class Chebyshev(Poly):
	"""
	Chebyshev polynomials of the fist kind
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
		recurrence = lambda p1, p2, n, x: self._backend.multiply(x, p1)*2 - p2
		Poly.__init__(self, self._backend, x, degree, initial, recurrence, **kw)

class Chebyshev2(Poly):
	"""
	Chebyshev polynomials of the second kind
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
		recurrence = lambda p1, p2, n, x: self._backend.multiply(x, p1)*2 - p2
		Poly.__init__(self, self._backend, x, degree, initial, recurrence, **kw)