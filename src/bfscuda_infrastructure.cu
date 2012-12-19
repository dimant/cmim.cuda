#include "bfscuda_device.h"
#include "bfscuda_local.h"
#include "cuda_utils.h"


thrust::device_vector<int>* sort_buffer;
int* raw_ptr_sort_buffer;
thrust::device_vector<int>* sort_buffer_sequence_id;
int* raw_ptr_sequence_id;

thrust::device_vector<int>* sort_buffer_one;
thrust::device_vector<int>* scanned_counts;

thrust::device_vector<int>* stencil;
thrust::device_vector<int>* const_ones;

thrust::device_vector<int>* d_unique_tuples;
thrust::device_vector<int>* d_tuple_counts;	

thrust::host_vector<int>* h_unique_tuples;
thrust::host_vector<int>* h_tuple_counts;	

int* raw_ptr_d_vars;
int* stride_vars;

int init_duplicate_counter(Data &d, int parallelism) {
	int sort_buffer_one_n = d.p * 2;
	int sort_buffer_n = sort_buffer_one_n * parallelism;

	sort_buffer = new thrust::device_vector<int>(sort_buffer_n);
	raw_ptr_sort_buffer = thrust::raw_pointer_cast(&(*sort_buffer)[0]);

	sort_buffer_sequence_id = new thrust::device_vector<int>(sort_buffer_n);
	raw_ptr_sequence_id = thrust::raw_pointer_cast(&(*sort_buffer_sequence_id)[0]);

	sort_buffer_one = new thrust::device_vector<int>(sort_buffer_one_n);
	scanned_counts = new thrust::device_vector<int>(sort_buffer_one_n);
	
	stencil = new thrust::device_vector<int>(sort_buffer_one_n + 1);
	const_ones = new thrust::device_vector<int>(sort_buffer_one_n + 1);
	
	d_unique_tuples = new thrust::device_vector<int>(sort_buffer_one_n);
	d_tuple_counts = new thrust::device_vector<int>(sort_buffer_one_n);

	h_unique_tuples = new thrust::host_vector<int>(sort_buffer_one_n);
	h_tuple_counts = new thrust::host_vector<int>(sort_buffer_one_n);

	// be careful about the width of the array
	// if d.q is less than 2 x n_vars you will run into problems
	CUDA_SAFE_CALL(cudaMalloc(&raw_ptr_d_vars, sizeof(int) * parallelism * d.q * 2));
	stride_vars = (int*) malloc(sizeof(int) * parallelism * d.q * 2);

	return 0;
}

int release_duplicate_counter(){
	delete sort_buffer;
	delete sort_buffer_sequence_id;
	delete sort_buffer_one;
	delete scanned_counts;
	delete stencil;
	delete const_ones;
	delete d_unique_tuples;
	delete d_tuple_counts;
	delete h_unique_tuples;
	delete h_tuple_counts;
	CUDA_SAFE_CALL(cudaFree(raw_ptr_d_vars));
	free(stride_vars);

	return 0;
}