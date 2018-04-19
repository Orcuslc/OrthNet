from ..backend import NumpyBackend, TensorflowBackend, TorchBackend

from .polynomial import Poly

class Jacobi(Poly):
	"""
	Jacobi polynomials
	"""
	def __init__(self, x, degree, alpha, beta, *args, **kw):
		"""
		input:
			- x: a tensor
			- degree: highest degree of polynomial
			- alpha, beta: parameters for Jacobi polynomials
		"""
		self._all_backends = list(filter(lambda backend: backend.is_available(), [TensorflowBackend(), TorchBackend(), NumpyBackend()]))
		self._backend = None
		for backend in self._all_backends:
			if backend.is_compatible(x):
				self._backend = backend
				break
		if self._backend is None:
			raise TypeError("Cannot determine backend from input arguments of type `{1}`. Available backends are {2}".format(type(self.x), ", ".join([str(backend) for backend in self._all_backends])))
		initial = [lambda x: self._backend.ones_like(x), lambda x: x*0.5*(alpha+beta+2)+0.5*(alpha-beta)]
		recurrence = lambda p1, p2, n, x: (self._backend.multiply(x*(2*n+alpha+beta)*(2*n+alpha+beta-2)+alpha**2-beta**2, p1)*(2*n+alpha+beta-1) - p2*(n+alpha-1)*(n+beta-1)*(2*n+alpha+beta)*2)/(2*n*(n+alpha+beta)*(2*n+alpha+beta-2))
		Poly.__init__(self, self._backend, x, degree, initial, recurrence, *args, **kw)