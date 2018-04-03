from distutils.core import setup, Extension

ext = Extension('orthnet.utils._enum_dim',
	sources = ['orthnet/utils/enum_dim/enum_dim.i'],
	language = 'c++',
	swig_opts = ['-c++'],
	extra_compile_args = [
		'-std=c++11',
		'-Wall',
		'-Ofast',
	]
)

setup(
	name = 'orthnet',
	version = '0.2.2',
	keywords = ['orthogonal polynomial', 'tensorflow', 'pytorch'],
	description = 'TensorFlow and PyTorch layers for generating orthogonal polynomials',
	license = 'MIT',
	author = 'Chuan Lu',
	author_email = 'chuan-lu@uiowa.edu',
	ext_modules = [ext],
	packages = ['orthnet', 'orthnet.poly', 'orthnet.utils'],
	)