#/bin/bash
swig -python -c++ enum_dim.i
g++ -c -fPIC enum_dim_wrap.cxx -I/usr/include/python3.6
g++ -shared enum_dim_wrap.o -o _enum_dim.so 
