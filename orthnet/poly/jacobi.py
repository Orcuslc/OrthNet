import tensorflow as tf
import torch
import numpy as np
from .poly import Poly

class Jacobi(Poly):
	"""
	Jacobi Polynomials
	"""
	def __init__(self, module, degree, x, alpha, beta, dtype = 'float32', loglevel = 0, index_comb = None):
		"""
		input:
			module: 'tensorflow', 'pytorch' or 'numpy'
			degree: highest degree of polynomial
			x: a tensor of shape [Nsample*Nparameter], each row is a sample point, each column represents a parameter
			alpha, beta: the parameters of Jacobi polynomials
			dtype: 'float32' or 'float64'
			loglevel: 1 to print time info and 0 to mute
			index_comb: combination of tensor product indices. If index_comb == None, the class will generate a new combination.
		"""
		if module == 'tensorflow':
			initial = [lambda x: tf.ones_like(x), lambda x: 0.5*(alpha+beta+2)*x+0.5*(alpha-beta)]
			recurrence = lambda p1, p2, n, x: ((2*n+alpha+beta-1)*tf.multiply((2*n+alpha+beta)*(2*n+alpha+beta-2)*x+alpha**2-beta**2, p1) - 2*(n+alpha-1)*(n+beta-1)*(2*n+alpha+beta)*p2)/(2*n*(n+alpha+beta)*(2*n+alpha+beta-2))
		elif module == 'pytorch':
			initial = [lambda x: torch.ones_like(x), lambda x: 0.5*(alpha+beta+2)*x+0.5*(alpha-beta)]
			recurrence = lambda p1, p2, n, x: ((2*n+alpha+beta-1)*((2*n+alpha+beta)*(2*n+alpha+beta-2)*x+alpha**2-beta**2)*p1 - 2*(n+alpha-1)*(n+beta-1)*(2*n+alpha+beta)*p2)/(2*n*(n+alpha+beta)*(2*n+alpha+beta-2))
		elif module == 'numpy':
			initial = [lambda x: np.ones_like(x), lambda x: 0.5*(alpha+beta+2)*x+0.5*(alpha-beta)]
			recurrence = lambda p1, p2, n, x: ((2*n+alpha+beta-1)*((2*n+alpha+beta)*(2*n+alpha+beta-2)*x+alpha**2-beta**2)*p1 - 2*(n+alpha-1)*(n+beta-1)*(2*n+alpha+beta)*p2)/(2*n*(n+alpha+beta)*(2*n+alpha+beta-2))			
		Poly.__init__(self, module, degree, x, initial, recurrence, dtype, loglevel, index_comb)