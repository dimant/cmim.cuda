#include "bfscuda_local.h"

__global__ void selectVariables(
								int* sort_buffer, int* sequence_ids, 
								int* d_data, int* d_vars,
								int n, int n_vars, int pitch, int offset, 
								int vars_width, int y_width, int sort_buffer_n) {


	int row_index = blockIdx.x * blockDim.x + threadIdx.x + offset;
	int col_index = blockIdx.y;

	int tmp = row_index / n;
	int sequence_id = tmp / 2;

	int shl_width_raw = d_vars[sequence_id * n_vars * 2 + col_index * 2];

	if(row_index < sort_buffer_n && (0x80000000 & shl_width_raw)) {

		int write_y = tmp & 0x00000001;

		int shl_width = (0x7FFFFFFF & shl_width_raw);
		int var_index = d_vars[sequence_id * n_vars * 2 + col_index * 2 + 1];

		int projection = (d_data[(row_index % n) * pitch + var_index] << shl_width);
		projection >>= (1 - write_y) * y_width;
		projection |= (0x01 << vars_width) * write_y;

		atomicOr(&sort_buffer[row_index], projection);
		sequence_ids[row_index] = sequence_id;
	}
}

