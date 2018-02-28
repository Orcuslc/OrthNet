%module enum_dim
%{
	#include "enum_dim.h"
%}

%include "std_vector.i"
%template() std::vector<int>;
%template() std::vector<std::vector<int> >;

%include "enum_dim.h"