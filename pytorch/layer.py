import torch

def legendre_list(n, x):
	"""
	Generate 1-dimensional Legendre polynomial by three-term recursion:
		P_{n+1} = 1/(n+1)*((2n+1)xP_n - nP_{n-1})

	input:
		n: order of target polynomial
		x: tensor for evaluating function values

	output:
		y: a list of function values
	"""
	assert n >= 0, "Order should >= 0."
	assert isinstance(x, torch.autograd.Variable) or isinstance(x, torch.Tensor), "x should be a instance of torch.autograd.Variable or torch.Tensor."
	if n == 0:
		return [torch.ones_like(x)]
	elif n == 1:
		return [torch.ones_like(x), x]
	polys = [torch.ones_like(x), x]
	for i in range(1, n):
		polys.append(((2*i+1)*x*polys[-1]-i*polys[-2])/(i+1))
	return polys

def legendre_tensor(n, x):
	"""
	Generate 1-dimensional Legendre polynomial by three-term recursion:
		P_{n+1} = 1/(n+1)*((2n+1)xP_n - nP_{n-1})

	input:
		n: order of target polynomial
		x: tensor for evaluating function values

	output:
		y: a tensor of function values
	"""
	assert n >= 0, "Order should >= 0."
	assert isinstance(x, torch.autograd.Variable) or isinstance(x, torch.Tensor), "x should be a instance of torch.autograd.Variable or torch.Tensor."	
	if n == 0:
		return torch.ones_like(x)
	elif n == 1:
		return torch.cat([torch.ones_like(x), x], dim = 1)
	polys = [torch.ones_like(x), x]
	for i in range(1, n):
		polys.append(((2*i+1)*x*polys[-1]-i*polys[-2])/(i+1))
	return torch.cat(polys, dim = 1)