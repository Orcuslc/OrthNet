from __future__ import print_function
from ..backend import TensorflowBackend, TorchBackend, NumpyBackend


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
		return polys[0]
	for i in range(1, degree):
		polys.append(recurrence(polys[-1], polys[-2], i, x))
	return polys


class Poly(object):
	def __init__(self, x, degree, initial, recurrence):
		self.x = x
		self._degree = degree
		self._initial = initial
		self._recurrence = recurrence
		self._backend = None

		self._all_backends = list(filter(lambda backend: backend.is_available(), [TensorflowBackend(), TorchBackend(), NumpyBackend()]))

	@property
	def backend(self):
		if self._backend is None:
			for backend in self._all_backends:
				if backend.is_compatible(self.x):
					self._backend = backend
					break
			if self._backend is None:
				raise TypeError("Cannot determine backend from input arguments of type `{1}`. Available backends are {2}".format(type(self.x), ", ".join([str(backend) for backend in self._all_backends])))
		return self._backend

