import tensorflow as tf
from ..utils.multi_dim import enumerate_dim as enum_dim_py
from ..utils._enum_dim import enum_dim as enum_dim_cpp


class Poly1d:
	"""
	Base class, 1-dimensional orthogonal polynomials
	"""
	def __init__(self, n, x, initial, recurrence):
		"""
		- input:
			- n: order of highest polynomial
			- x: tensor for evaluating function values
			- initial: initial value, a list of two functions
			- recurrence: the function of recurrence, with three variables:
				P_{n+1} = f(P_{n}, P_{n-1}, n, x) 
			(initial and recurrence are functions on tensorflow.Tensor)
		"""
		assert n >= 0 and isinstance(n, int), "Order should be a non-negative integer."
		assert isinstance(x, tf.Variable) or isinstance(x, tf.Tensor), "x should be an isinstance of tensorflow.Variable or tensorflow.Tensor."
		assert len(initial) == 2, "Need two initial functions."

		self.n = n
		self.x = x
		self.initial = initial
		self.recurrence = recurrence
		self._compute_recurrence()

	def _compute_recurrence(self):
		