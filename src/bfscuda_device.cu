#include <iostream>
#include <vector>

#include <stdio.h>
#include <stdlib.h>

//#define THRUST_USE_MERRILL_RADIX_SORT

#include <thrust/iterator/constant_iterator.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/fill.h>

#include "bfscuda_device.h"
#include "bfscuda_local.h"
#include "cuda_utils.h"
#include "utils.h"

//__device__ __constant__ int d_vars[VAR_ARRAY_SIZE];


	/*unsigned int timer;
	timer = 0;
	CUT_SAFE_CALL(cutCreateTimer(&timer));
	CUT_SAFE_CALL(cutStartTimer(timer));*/

	
	/*CUT_SAFE_CALL(cutStopTimer(timer));
	cout << "total: " << cutGetTimerValue(timer) << endl;

	CUT_SAFE_CALL(cutDeleteTimer(timer));*/

struct set_neq_flag {
	__host__ __device__
	int operator() (int a, int b){
		return (a != b) ? 0x80000001 : 1;
	}
};

struct my_reduce_by_key {
	__host__ __device__
	int operator() (int a, int b){
		return ((b & 0x80000000) ? (b & 0x7FFFFFFF) : ((a & 0x7FFFFFFF) + (b & 0x7FFFFFFF)))
			| (a & 0x80000000) | (b & 0x80000000) ;
	}
};

struct my_copy_if_op {
	__host__ __device__
	int operator() (int a){
		return (a & 0x80000000) >> 31;
	}
};

int count_duplicates(Data &d, int parallelism, int vars_width, int** vars, thrust::host_vector< thrust::host_vector<int> > &h_counts) {
	int i, j, k;

	int sort_buffer_one_n = d.p * 2;
	int sort_buffer_n = sort_buffer_one_n * parallelism;

	int stride_vars_n = parallelism * d.q * 2;

	cudaError_t err;

	memset(stride_vars, 0, sizeof(int) * stride_vars_n);

	
	k = 0;
	int shl_width;
	int n_vars = 0;
	int total_width = 0;
	for(i = 0; i < parallelism; i++) {		
		shl_width = 0;
		if(n_vars < vars[i][0])
			n_vars = vars[i][0];
		for(j = 1; j < vars[i][0] * 2 + 1; j+=2) {
			stride_vars[k++] = 0x80000000 | shl_width;
			stride_vars[k++] = vars[i][j + 1];

			shl_width += vars[i][j];

			//std::cout <<  (0x7fffffff & stride_vars[k - 2]) << " ";
		}
		//std::cout << std::endl;

		if(total_width < shl_width)
			total_width = shl_width;
		
		if(j < vars_width - 1) {
			stride_vars[k++] = 0;
			stride_vars[k++] = 0;
		}
	}
	
	//std::cout << std::endl;
	//for(i = 0; i < parallelism; i++) {
	//	for(j = 0; j < vars[i][0] * 2 + 1; j++) 
	//		std::cout << vars[i][j] <<  " ";
	//	std::cout << std::endl;
	//}
	//
	//std::cout << std::endl;
	//for(i = 0; i < k; i++) {
	//	if(i % 2 == 0)
	//		std::cout << (0x7fffffff & stride_vars[i]) << " ";
	//	else
	//		std::cout << stride_vars[i] << " ";
	//}
	//std::cout << std::endl << n_vars << std::endl;
	
	thrust::fill(sort_buffer->begin(), sort_buffer->begin() + sort_buffer_n, 0);

	CUDA_SAFE_CALL(cudaMemcpy(raw_ptr_d_vars, stride_vars, sizeof(int) * stride_vars_n, cudaMemcpyHostToDevice));

	struct cudaDeviceProp* prop = getDeviceProperties();

	if(prop->maxGridSize[1] < d.q) {
		Logger& l = Logger::getInstance();
		l.error("Code needs fixing to work with a data set that wide.\n");
		return 1;
	}

	int block_size_x = prop->maxThreadsDim[0];

	int total_block_count_x = (sort_buffer_n / block_size_x) + (sort_buffer_n % block_size_x > 0 ? 1 : 0);
	int block_count_y = n_vars;

	int block_count_x = (total_block_count_x <= prop->maxGridSize[0] ? total_block_count_x : prop->maxGridSize[0]);

	dim3 numBlocks(block_count_x, block_count_y);
	dim3 threadsPerBlock(block_size_x);


	int * raw_ptr_sort_buffer_iter = raw_ptr_sort_buffer;
	int * raw_ptr_sequence_id_iter = raw_ptr_sequence_id;

	for(i = total_block_count_x; i > 0; i -= block_count_x) {
		if(i < 0)
			numBlocks.x = block_count_x + i;

		selectVariables<<<numBlocks, threadsPerBlock>>>(
			raw_ptr_sort_buffer_iter, 
			raw_ptr_sequence_id_iter,
			d.raw_ptr_data, raw_ptr_d_vars,
			d.p, n_vars, d.d_pitch / sizeof(int), raw_ptr_sort_buffer_iter - raw_ptr_sort_buffer, 
			total_width, vars[0][1], sort_buffer_n);

		err = cudaGetLastError();
		if(err != cudaSuccess) {
			Logger& l = Logger::getInstance();
			std::stringstream output;
			output << "CUDA error code: " << err << " - " << cudaGetErrorString(err) << std::endl;
			l.error(output.str());
			return 1;
		}

		raw_ptr_sort_buffer_iter += block_count_x * block_size_x;
		raw_ptr_sequence_id_iter += block_count_x * block_size_x;
		
	}

	
	//for(i = 0; i < sort_buffer->size(); i++) {
	//	std::cout << (*sort_buffer_sequence_id)[i] << ", " << i << ": ";
	//	print_binary_int((*sort_buffer)[i]);
	//}
	
	//std::cout << "Y width: " << vars[0][1] << std::endl;

	//int* h_data;
	//cudaMallocHost((void**) &h_data, sizeof(int) * d.p * d.q);
	//cudaMemcpy2D((void**) h_data, sizeof(int) * d.q, d.raw_ptr_data, d.d_pitch, sizeof(int) * d.q, d.p, cudaMemcpyDeviceToHost);

	//int* h_vars;
	//cudaMallocHost((void**) &h_vars, sizeof(int) * parallelism * d.q);
	//cudaMemcpy(h_vars, raw_ptr_d_vars, sizeof(int) * parallelism * d.q, cudaMemcpyDeviceToHost);

	//for(i = 0; i < 4 * sort_buffer_one_n; i++) {
	//	for(j = 0; j < n_vars * 2; j++) {
	//		if(0x80000000 & h_vars[(i / sort_buffer_one_n) * n_vars * 2 + j])
	//			std::cout << "x";
	//		std::cout << (0x7FFFFFFF & h_vars[(i / sort_buffer_one_n) * n_vars * 2 + j]) << " ";
	//	}
	//	std::cout << std::endl;

	//	std::cout << i << " " << std::endl;
	//	for(j = 2; j < vars[i / sort_buffer_one_n][0] * 2 + 1; j+=2) {
	//		std::cout << vars[i / sort_buffer_one_n][j-1] << ": ";
	//		print_binary_int(h_data[(i % d.p) * d.q + vars[i / sort_buffer_one_n][j]]); std::cout << std::endl;
	//	}
	//	std::cout << (*sort_buffer)[i] << ": ";
	//	print_binary_int((*sort_buffer)[i]);
	//	std::cout << std::endl <<  "---------------------" << std::endl;
	//}
	//exit(0);

	// find duplicates
	thrust::sort_by_key(
		sort_buffer->begin(), 
		sort_buffer->begin() + sort_buffer_n, 
		sort_buffer_sequence_id->begin());

	thrust::stable_sort_by_key(
		sort_buffer_sequence_id->begin(), 
		sort_buffer_sequence_id->begin() + sort_buffer_n,
		sort_buffer->begin());

	for(i = 0; i < parallelism; i++) {
		(*stencil)[0] = 1;
		(*stencil)[sort_buffer_one_n] = 1;
		(*const_ones)[0] = 0x80000001;
		(*const_ones)[sort_buffer_one_n] = 0x80000001;

		// compute stencil
		thrust::transform(
			sort_buffer->begin() + i * sort_buffer_one_n, 
			sort_buffer->begin() + (i + 1) * sort_buffer_one_n - 1, 
			sort_buffer->begin() + i * sort_buffer_one_n + 1,
			const_ones->begin() + 1, 
			set_neq_flag());

		// count duplicates
		thrust::inclusive_scan( 
			const_ones->begin(),
			const_ones->end() - 1,
			scanned_counts->begin(),
			my_reduce_by_key());

		thrust::fill(d_unique_tuples->begin(), d_unique_tuples->end(), 0);
		thrust::fill(d_tuple_counts->begin(), d_tuple_counts->end(), 0);

		thrust::copy_if(
			thrust::make_zip_iterator(
				thrust::make_tuple(
					sort_buffer->begin() + i * sort_buffer_one_n,
					scanned_counts->begin())),
			thrust::make_zip_iterator(
				thrust::make_tuple(
					sort_buffer->begin() + i * sort_buffer_one_n + sort_buffer_one_n,
					scanned_counts->end())),
			const_ones->begin() + 1,
			thrust::make_zip_iterator(
				thrust::make_tuple(
					d_unique_tuples->begin(),
					d_tuple_counts->begin())),
			my_copy_if_op());

		thrust::copy(
			thrust::make_zip_iterator(
				thrust::make_tuple(
					d_unique_tuples->begin(),
					d_tuple_counts->begin())),
			thrust::make_zip_iterator(
				thrust::make_tuple(
					d_unique_tuples->end(),
					d_tuple_counts->end())),
			thrust::make_zip_iterator(
				thrust::make_tuple(
					h_counts[i * 2].begin(),
					h_counts[i * 2 + 1].begin())));
	}

	return 0;
}
