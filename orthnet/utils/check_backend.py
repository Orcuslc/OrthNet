from ..backend import NumpyBackend, TensorflowBackend, TorchBackend

def check_backend(x):
	all_backends = list(filter(lambda backend: backend.is_available(), [TensorflowBackend(), TorchBackend(), NumpyBackend()]))
	for backend in _all_backends:
		if backend.is_compatible(x):
			return backend
	raise TypeError("Cannot determine backend from input arguments of type `{1}`. Available backends are {2}".format(type(self.x), ", ".join([str(backend) for backend in _all_backends])))