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
		a 2D-list, each element a combination (a list)
*/
	std::vector<std::vector<int> > res;
	std::vector<int> cur;
	dfs(res, cur, n, dim);
	return res;
}