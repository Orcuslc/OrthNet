"""
numpy backend
"""
try:
	import numpy as np
except ImportError:
	_np = None

from .backend import Backend, assert_backend_available


class NumpyBackend(Backend):

	def __str__(self):
		return "numpy"

	def is_available(self):
		return _np is not None

	@assert_backend_available
	def is_compatible(self, args):
		assert list(filter(lambda t: isinstance(args, t), [
				np.ndarray,
				np.matrix
			])) != [], "numpy backend requires input to be an instance of `np.ndarray` or `np.matrix`"
		return True