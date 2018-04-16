from __future__ import print_function
from ..backend import TensorflowBackend, TorchBackend, NumpyBackend


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

	