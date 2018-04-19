"""
tensorflow backend
"""
try:
	import tensorflow as tf
except ImportError:
	tf = None

from ._backend import Backend, assert_backend_available


class TensorflowBackend(Backend):

	def __str__(self):
		return "tensorflow"

	def is_available(self):
		return tf is not None

	@assert_backend_available
	def is_compatible(self, args):
		if list(filter(lambda t: isinstance(args, t), [
				tf.Tensor,
				tf.Variable
			])) != []:
			return True
			# "tensorflow backend requires input to be an isinstance of `tensorflow.Tensor` or `tensorflow.Variable`"
		return False

	def concatenate(self, tensor, axis):
		return tf.concat(tensor, axis = axis)

	def ones_like(self, tensor):
		return tf.ones_like(tensor)

	def multiply(self, x, y):
		return tf.multiply(x, y)

	def expand_dims(self, tensor, axis):
		return tf.expand_dims(tensor, axis)

	def get_dims(self, tensor):
		return [dim.value for dim in tensor.get_shape()]

	def reshape(self, tensor, shape):
		return tf.reshape(tensor, shape)

	def matmul(self, tensor1, tensor2):
		return tf.matmul(tensor1, tensor2)
