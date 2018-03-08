#include <vector>

void dfs(std::vector<std::vector<int> >& res, std::vector<int> cur, int n, int dim) {
	if(dim == 1) {
		cur.push_back(n);
		res.push_back(cur);
		return;
	}
	for(int i = n; i >= 0; i--) {
		cur.push_back(i);
		dfs(res, cur, n-i, dim-1);
		cur.pop_back();
	}
}

std::vector<std::vector<int> > enum_dim(int n, int dim) {
/*
	enumerate dims;

	input:
		n: total order
		dim: dimension (number of variables)

	output:
		a 2D-list, the first element is the index of each degree, others a combination (a tuple)
*/
	std::vector<std::vector<int> > res;
	res.push_back(std::vector<int>()); 
	for(int i = 0; i <= n; i++) {
		res[0].push_back(res.size()-1);
		dfs(res, std::vector<int>(), i, dim);
	}
	return res;
}