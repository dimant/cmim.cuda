#define BFSCUDA_EXPORTS

#include <cstdlib>

#include "bfscuda.h"
#include "r_binding.h"
#include "utils.h"
#include "cuda_utils.h"

#include "mrmr.h"
#include "mrmr_data.h"

class RRowIterator : public RowIterator {
private:
	char** data;
	int n;
	int p;
	int rowIdx;
public:
	RRowIterator(char** data, int n, int p) :data(data), n(n), p(p), rowIdx(0) {
	};

	void get_row(std::vector<std::string>& row);
};

void RRowIterator::get_row(std::vector<std::string>& row) {
	int i;

	row.clear();

	if(rowIdx < n) {
		for(i = rowIdx * p; i < (rowIdx + 1) * p; i++) {
			row.push_back(std::string(data[i]));
		}
	}
	
	rowIdx++;
}

void r_BFSCuda(char** data, int* n, int* p, int* k, int* max_parallelism, double* weights, double* indices) {
	RRowIterator ri(data, *n, *p);
	Data d;
	std::vector<struct result_record*> all_results;
	std::vector<struct result_record*> best_results;
	int i;
	Logger& l = Logger::getInstance();

	loadData(d, ri);

	process(d, *k, *max_parallelism, all_results);
	
	backward_search(d, *k, all_results, best_results);

	for(i = 0; i < *p - 1; i++) {
		weights[i] = best_results[i]->i_u_yx;
		indices[i] = best_results[i]->vars[4] + 1;
		free(best_results[i]);
	}

	cudaError_t state = cudaFree(d.raw_ptr_data); 
	if(state != cudaSuccess) { 
		l.error(std::string(cudaGetErrorString(state)) + "\n"); 
		return;
	}	

	return;
}

void r_mifsu2(char** data, int* n, int* p, int* max_parallelism, double* weights, double* a, double* b) {
	RRowIterator ri(data, *n, *p);
	Data d;
	std::vector<struct result_record*> all_results;

	loadData(d, ri);

}

void r_mRRmatrix(char** data, int* n, int* p, int* max_parallelism, double* matrix) {
	RRowIterator ri(data, *n, *p);
	Data d;
	std::vector<struct result_record*> all_results;
	std::vector<struct result_record*> best_results;

	int i;
	int ncol = *p - 1;
	int xi, yi;
		
	loadData(d, ri);
	process(d, 1, *max_parallelism, all_results);

	for(i = 0; i < all_results.size(); i++) {
		xi = all_results[i]->vars[4];
		yi = all_results[i]->vars[6];
		matrix[xi * ncol + yi] = all_results[i]->i_u_yx;

		free(all_results[i]);
	}

	double weight;

	for(xi = 0; xi < ncol; xi++) {
		for(yi = xi + 1; yi < ncol; yi++) {
			weight = matrix[yi * ncol + xi] + matrix[xi * ncol + yi];
			matrix[yi * ncol + xi] = weight;
			matrix[xi * ncol + yi] = weight;
		}
	}

	return;
}

void r_HardwareSupport(char** name, int* memory, int* major, int* minor, char** status) {
	struct cudaDeviceProp* prop = getDeviceProperties();

	// do we have a cuda device?
	// does it have compute capability >= 1.1?
	if(prop == 0 || 
		prop->major < 1 || 
		(prop->major == 1 && prop->minor < 1)) {
			*status = strdup("Not OK.");
	} else {
		*name = strdup(prop->name);
		*memory = prop->totalGlobalMem;
		*major = prop->major;
		*minor = prop->minor;
		*status = strdup("OK");
	}

	return;
}


void readData(DataTable* myData, char** data, int* n, int* p) {
  int i, j;
  float tmp;
  
  myData->data = new float[(*n) * (*p)];
  myData->nsample = *n;
  myData->nvar = *p;

  float* data_cursor = myData->data;

  for(i = 0; i < *n; i++) {
	for(j = 0; j < *p; j++) {
	  *data_cursor = (float) atof(data[i * (*p) + j]);
	}

	// the MRMR code expects the class label to be the first column
	// however within the CUDA.CMIM framework we expect the class label
	// to be last - hence the conversion
	tmp = myData->data[i * (*p)];
	myData->data[i * (*p)] = myData->data[i * (*p) + (*p) - 1];
	myData->data[i * (*p) + (*p) - 1] = tmp;
  }

}

void r_MRMR(char** data, int* n, int* p, double* indices) {
	int i;
	DataTable* myData = new DataTable;
	long* l_indices;

	int select_method = MID;

	readData(myData, data, n, p);
	myData->discretize(9999, 0); // no discretization, only pre-processing

	l_indices = mRMR_selection(myData, myData->nvar, FeaSelectionMethod(select_method));

	for(i = 0; i < myData->nvar; i++) {
		indices[i] = (double) l_indices[i];
	}

	delete l_indices;
}