"""
torch backend
"""
try:
	import torch
except ImportError:
	_torch = None

from .backend import Backend, assert_backend_available


class TorchBackend(Backend):

	def __str__(self):
		return "torch"

	def is_available(self):
		return _torch is not None

	@assert_backend_available
	def is_compatible(self, args):
		assert list(filter(lambda t: isinstance(args, t), [
				torch.FloatTensor,
				torch.DoubleTensor,
				torch.cuda.FloatTensor,
				torch.cuda.DoubleTensor
			])) != [], "torch backend requires input to be an instance of `torch.FloatTensor`, `torch.DoubleTensor`, `torch.cuda.FloatTensor` or `torch.cuda.DoubleTensor`"
		return True

	def concatenate(self, tensor, axis):
		return torch.cat(tensor, dim = axis)