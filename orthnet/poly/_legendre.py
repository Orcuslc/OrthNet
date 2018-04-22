from ..utils import check_backend
from .polynomial import Poly
from math import sqrt

class Legendre(Poly):
	"""
	Legendre Polynomials
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
		recurrence = lambda p1, p2, n, x: (self._backend.multiply(x, p1)*(2*n+1)-p2*n)/(n+1)
		Poly.__init__(self, self._backend, x, degree, initial, recurrence, **kw)


class Legendre_Normalized(Poly):
	"""
	Normalized Legendre Polynomials with inner product be 1 if n == m.
	"""
	def __init__(self, x, degree, backend = None, **kw):
		"""
		input:
			- x: a tensor
			- degree: highest degree of polynomial
		"""
		if backend is None:
			self._backend  = check_backend(x)
		else:
			self._backend = backend
		initial = [lambda x: self._backend.ones_like(x)*sqrt(1/2), lambda x: x*sqrt(3/2)]
		recurrence = lambda p1, p2, n, x: (self._backend.multiply(x, p1)*sqrt((2*n+1)*(2*n+3))-p2*n*sqrt((2*n+3)/(2*n-1)))/(n+1)
		Poly.__init__(self, self._backend, x, degree, initial, recurrence, **kw)