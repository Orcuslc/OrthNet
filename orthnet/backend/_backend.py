from functools import wraps

def assert_backend_available(f):
	@wraps(f)
	def check(backend, *args, **kw):
		if not backend.is_available:
			raise RuntimeError(
				"Backend `{1}` is not available".format(str(backend)))
		return f(backend, *args, **kw)
	return check


class Backend(object):
	def __str__(self):
		return "<backend>"

	def __false(self):
		return False

	is_available = is_compatible = __false

	def concatenate(self, tensor, axis):
		return None

	def ones_like(self, tensor):
		return None

	def multiply(self, x, y):
		return None

	def expand_dims(self, tensor, axis):
		return None

	def get_dims(self, tensor):
		return None

	def reshape(self, tensor, shape):
		return None

	def matmul(self, tensor1, tensor2):
		return None