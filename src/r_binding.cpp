#define BFSCUDA_EXPORTS

#include <cstdlib>
#include <algorithm>

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

void r_cmifs_exact(char** data, int* n, int* p, int* k, int* max_parallelism, double* infgain, double* weights, double* indices) {
	RRowIterator ri(data, *n, *p);
	Data d;
	std::vector<struct result_record*> all_results;
	std::vector<struct result_record*> best_results;
	int i;
	Logger& l = Logger::getInstance();

	loadData(d, ri);

	process(d, *k, *max_parallelism, all_results);

	for(i = 0; i < all_results.size(); i++) {
		all_results[i]->i_u_yx = all_results[i]->i_u_yx / infgain[all_results[i]->vars[4]];
	}
	
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

bool is_sister_edge(struct result_record* xi, struct result_record* xj) {
	int i;
	int j;

	int k1 = xi->vars[0];
	int k2 = xj->vars[0];
	int k = (k1 < k2) ? k1 : k2;

	bool is_element = false;
	bool same_edge = true;
	bool result = true;
	bool eql;

	for(i = 4; i < 4 + (k-1) * 2; i += 2) {
		is_element = false;

		for(j = 4; j < 4 + (k-1) * 2; j += 2) {
			eql = (xi->vars[i] == xj->vars[j]);
			is_element |= eql;
			if(i == j)
				same_edge &= eql;
		}

		result &= is_element;
	}

	result &= (same_edge == false);

	return result;
}

void r_mifsmed(char** data, int* n, int* p, int* k, int* max_parallelism, double* weights) {
	RRowIterator ri(data, *n, *p);
	Data d;
	std::vector<struct result_record*> all_results;
	int h, i, j;

	loadData(d, ri);
	process(d, *k, *max_parallelism, all_results);

	std::vector<struct result_record*> edges;
	std::vector<int> mark_for_removal;
	double edge_weight;
	std::vector<double> edge_weights;

	for(h = 0; h < *p; h++) {
		edges.clear();

		for(i = 0; i < all_results.size(); i++) {
			for(j = 4; j < 4 + (all_results[i]->vars[0] - 1) * 2; j += 2) {
				if(all_results[i]->vars[j] == h)
					edges.push_back(all_results[i]);
			}
		}

		while(edges.empty() == false) {
			mark_for_removal.clear();

			
			edge_weight = edges[0]->i_u_yx;
			mark_for_removal.push_back(0);

			for(i = 1; i < edges.size(); i++) {
				if(is_sister_edge(edges[0], edges[i])) {
					edge_weight += edges[i]->i_u_yx;
					mark_for_removal.push_back(i);
				}
			}

 			for(i = 0; i < mark_for_removal.size(); i++) {
				edges.erase(edges.begin() + mark_for_removal[i] - i);
				if(edges.size() == 1) {
					edges.clear();
					break;
				}
			}

			edge_weights.push_back(edge_weight);
		}

		std::sort(edge_weights.begin(), edge_weights.end());
		
		int len = edge_weights.size();
		if(len % 2 == 0) {
			double top = edge_weights[len/2];
			double bottom = edge_weights[(len/2) - 1];

			weights[h] = bottom + ((top - bottom) / 2);
		} else {
			weights[h] = edge_weights[len/2];
		}
	}
}

void r_mifsuk(char** data, int* n, int* p, int* k, int* max_parallelism, double* weights) {
	RRowIterator ri(data, *n, *p);
	Data d;
	std::vector<struct result_record*> all_results;
	int i;
	int xi;

	loadData(d, ri);
	process(d, *k, *max_parallelism, all_results);

	for(i = 0; i < *p; i++) {
		weights[i] = 0.0;
	}

	for(i = 0; i < all_results.size(); i++) {
		xi = all_results[i]->vars[4];
		
		weights[xi] += all_results[i]->i_u_yx;
	}


	return;
}

void r_mRRmatrix(char** data, int* n, int* p, int* max_parallelism, double* matrix) {
	RRowIterator ri(data, *n, *p);
	Data d;
	std::vector<struct result_record*> all_results;
	Logger& l = Logger::getInstance();
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

	cudaError_t state = cudaFree(d.raw_ptr_data); 
	if(state != cudaSuccess) { 
		l.error(std::string(cudaGetErrorString(state)) + "\n"); 
		return;
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


//void readData(DataTable* myData, char** data, int* n, int* p) {
//  int i, j;
//  float tmp;
//  
//  myData->data = new float[(*n) * (*p)];
//  myData->nsample = *n;
//  myData->nvar = *p;
//
//  float* data_cursor = myData->data;
//
//  for(i = 0; i < *n; i++) {
//	for(j = 0; j < *p; j++) {
//	  *data_cursor = (float) atof(data[i * (*p) + j]);
//	}
//
//	// the MRMR code expects the class label to be the first column
//	// however within the CUDA.CMIM framework we expect the class label
//	// to be last - hence the conversion
//	tmp = myData->data[i * (*p)];
//	myData->data[i * (*p)] = myData->data[i * (*p) + (*p) - 1];
//	myData->data[i * (*p) + (*p) - 1] = tmp;
//  }
//
//}
//
//void r_MRMR(char** data, int* n, int* p, double* indices) {
//	int i;
//	DataTable* myData = new DataTable;
//	long* l_indices;
//
//	int select_method = MID;
//
//	readData(myData, data, n, p);
//	myData->discretize(9999, 0); // no discretization, only pre-processing
//
//	l_indices = mRMR_selection(myData, myData->nvar, FeaSelectionMethod(select_method));
//
//	for(i = 0; i < myData->nvar; i++) {
//		indices[i] = (double) l_indices[i];
//	}
//
//	delete l_indices;
//}