from ..utils import check_backend
from .polynomial import Poly

class Laguerre(Poly):
	"""
	Laguerre polynomials
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
		initial = [lambda x: self._backend.ones_like(x), lambda x: 1-x]
		recurrence = lambda p1, p2, n, x: (self._backend.multiply(p1, 2*n+1-x)-p2*n)/(n+1)
		Poly.__init__(self, self._backend, x, degree, initial, recurrence, **kw)