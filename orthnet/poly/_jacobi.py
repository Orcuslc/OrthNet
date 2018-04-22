from ..utils import check_backend
from .polynomial import Poly

class Jacobi(Poly):
	"""
	Jacobi polynomials
	"""
	def __init__(self, x, degree, alpha, beta, backend = None, **kw):
		"""
		input:
			- x: a tensor
			- degree: highest degree of polynomial
			- alpha, beta: parameters for Jacobi polynomials
		"""
		if backend is None:
			self._backend = check_backend(x)
		else:
			self._backend = backend
		initial = [lambda x: self._backend.ones_like(x), lambda x: x*0.5*(alpha+beta+2)+0.5*(alpha-beta)]
		recurrence = lambda p1, p2, n, x: (self._backend.multiply(x*(2*n+alpha+beta)*(2*n+alpha+beta-2)+alpha**2-beta**2, p1)*(2*n+alpha+beta-1) - p2*(n+alpha-1)*(n+beta-1)*(2*n+alpha+beta)*2)/(2*n*(n+alpha+beta)*(2*n+alpha+beta-2))
		Poly.__init__(self, self._backend, x, degree, initial, recurrence, **kw)