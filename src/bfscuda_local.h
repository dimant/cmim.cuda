#ifndef BFSCUDA_LOCAL
#define BFSCUDA_LOCAL

#include "bfscuda_device.h"

// min of x and y but if x == 0 return y
// can be extended to return max of x or y but who cares
// the point of using this is to avoid branching
#define min(x, y) (y ^ ((x ^ y) & -(x < y && x > 0)))

extern thrust::device_vector<int>* sort_buffer;
extern int* raw_ptr_sort_buffer;
extern thrust::device_vector<int>* sort_buffer_sequence_id;
extern int* raw_ptr_sequence_id;

extern thrust::device_vector<int>* sort_buffer_one;
extern thrust::device_vector<int>* scanned_counts;

extern thrust::device_vector<int>* stencil;
extern thrust::device_vector<int>* const_ones;

extern thrust::device_vector<int>* d_unique_tuples;
extern thrust::device_vector<int>* d_tuple_counts;	

extern thrust::host_vector<int>* h_unique_tuples;
extern thrust::host_vector<int>* h_tuple_counts;	

extern int* raw_ptr_d_vars;
extern int* stride_vars;

struct cudaDeviceProp* getDeviceProperties();

__global__ void selectVariables(
								int* sort_buffer, int* sequence_ids, 
								int* d_data, int* d_vars,
								int n, int n_vars, int pitch, int offset, 
								int vars_width, int y_width, int sort_buffer_n);

#endif