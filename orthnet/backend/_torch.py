"""
torch backend
"""
try:
	import torch
except ImportError:
	torch = None

from ._backend import Backend, assert_backend_available


class TorchBackend(Backend):

	def __str__(self):
		return "torch"

	def is_available(self):
		return torch is not None

	@assert_backend_available
	def is_compatible(self, args):
		if list(filter(lambda t: isinstance(args, t), [
				torch.FloatTensor,
				torch.DoubleTensor,
				torch.cuda.FloatTensor,
				torch.cuda.DoubleTensor
			])) != []:
			return True
		# , "torch backend requires input to be an instance of `torch.FloatTensor`, `torch.DoubleTensor`, `torch.cuda.FloatTensor` or `torch.cuda.DoubleTensor`"
		return False

	def concatenate(self, tensor, axis):
		return torch.cat(tensor, dim = axis)

	def ones_like(self, tensor):
		return torch.ones_like(tensor)

	def multiply(self, x, y):
		return torch.mul(x, y)

	def expand_dims(self, tensor, axis):
		return tensor.unsqueeze(axis)

	def get_dims(self, tensor):
		return tensor.size()

	def reshape(self, tensor, shape):
		return tensor.view(shape)

	def matmul(self, tensor1, tensor2):
		return torch.matmul(tensor1, tensor2)