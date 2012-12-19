#ifndef BFSCUDA_H
#define BFSCUDA_H
#include <vector>
#include <string>

#include "portability.h"

#include "bfscuda_data.h"
#include "bfscuda_device.h"

#define INT_SIZE (sizeof(int) * 8)
#define VAR_ARRAY_SIZE 16

#define SAFE_CALL(x) {int error = x; if(error != 0) return error;}

class RowIterator {

public:
	virtual void get_row(std::vector<std::string>& row) = 0;
};

class LexSubset {
private:
	int first;
	int m;
	int h;

public:
	LexSubset(int k);

	int nextSubset(int* a, int n, int k);
};

struct result_record {
	double i_u_yx;
	int vars[VAR_ARRAY_SIZE];
};

int loadData(Data &d, RowIterator &ri);
int process(Data &d, int k, int max_parallelism, std::vector<struct result_record*>& all_results);
int backward_search(Data d, int cond_vars, 
					  std::vector<struct result_record*> all_results, 
					  std::vector<struct result_record*>& best_results);

double evaluate_term(thrust::host_vector<int> &unique_tuples, thrust::host_vector<int> &tuple_counts, int* vars, int n);


#endif
