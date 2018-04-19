"""
numpy backend
"""
try:
	import numpy as np
except ImportError:
	np = None

from ._backend import Backend, assert_backend_available


class NumpyBackend(Backend):

	def __str__(self):
		return "numpy"

	def is_available(self):
		return np is not None

	@assert_backend_available
	def is_compatible(self, args):
		if list(filter(lambda t: isinstance(args, t), [
				np.ndarray,
				np.matrix
			])) != []:
			return True
		# , "numpy backend requires input to be an instance of `np.ndarray` or `np.matrix`"
		return False

	def concatenate(self, tensor, axis):
		return np.concatenate(tensor, axis = axis)

	def ones_like(self, tensor):
		return np.ones_like(tensor)

	def multiply(self, x, y):
		return x*y

	def expand_dims(self, tensor, axis):
		return np.expand_dims(tensor, axis)

	def get_dims(self, tensor):
		return tensor.shape

	def reshape(self, tensor, shape):
		return np.reshape(tensor, shape)

	def matmul(self, tensor1, tensor2):
		return np.dot(tensor1, tensor2)