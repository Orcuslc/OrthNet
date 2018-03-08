from _enum_dim import enum_dim as enum_dim_cpp
from multi_dim import enum_dim as enum_dim_recursive
from multi_dim import dim as enum_dim_py

for i in [3, 5, 7, 9]:
	for j in [10, 20, 30, 40, 50]:
		print(i, j)
		%timeit enum_dim_cpp(i, j)
		%timeit enum_dim_py(i, j)       
		%timeit enum_dim_recursive(i, j)
