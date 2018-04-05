import time
from functools import wraps

def timeit(loglevel):
	def _timeit(func):
		@wraps(func)
		def timed(*args, **kw):
			t1 = time.time()
			res = func(*args, **kw)
			print(func.__name__, (time.time() - t1))
			return res
		def untimed(*args, **kw):
			return func(*args, **kw)
		if loglevel == 0:
			return untimed
		elif loglevel == 1:
			return timed
	return _timeit