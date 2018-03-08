#/bin/bash
swig -python -c++ enum_dim.i
g++ -Ofast -c -fPIC enum_dim_wrap.cxx -I/usr/include/python3.5
g++ -Ofast -shared enum_dim_wrap.o -o _enum_dim.so 