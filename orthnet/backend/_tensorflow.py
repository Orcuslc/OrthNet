"""
tensorflow backend
"""
try:
	import tensorflow as tf
except ImportError:
	_tf = None

from .backend import Backend, assert_backend_available


class TensorflowBackend(Backend):

	def __str__(self):
		return "tensorflow"

	def is_available(self):
		return tf is not None

	@assert_backend_available
	def is_compatible(self, args):
		assert list(filter(lambda t: isinstance(args, t), [
				tf.Tensor,
				tf.Variable
			])) != [], "tensorflow backend requires input to be an isinstance of `tensorflow.Tensor` or `tensorflow.Variable`"
		return True

	def concatenate(self, tensor, axis):
		return tf.concat(tensor, axis = axis)