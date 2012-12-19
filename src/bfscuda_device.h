#ifndef BFSCUDA_DEVICE
#define BFSCUDA_DEVICE

#include "bfscuda_data.h"

int init_duplicate_counter(Data &d, int parallelism);
int release_duplicate_counter();
int count_duplicates(Data &d, int parallelism, int vars_width, int** vars, thrust::host_vector< thrust::host_vector<int> > &h_count_maps);

#endif