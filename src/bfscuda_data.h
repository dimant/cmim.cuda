#ifndef BFSCUDA_DATA
#define BFSCUDA_DATA

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct Data {
	size_t p;
	size_t q;
	int total_bits;
	size_t d_pitch;

	thrust::host_vector<int> variable_widths;
	thrust::host_vector<int> h_data;
	int* raw_ptr_data;
};

#endif