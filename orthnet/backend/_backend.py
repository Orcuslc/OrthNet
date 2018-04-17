from functools import wraps

def assert_backend_availale(f):
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
		pass

