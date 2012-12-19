#include "cuda_utils.h"
#include "bfscuda.h"

#include <cuda_runtime.h>


struct cudaDeviceProp* getDeviceProperties() {
	static struct cudaDeviceProp* prop = 0;
	
	if(prop == 0) {

		prop = (struct cudaDeviceProp*) malloc(sizeof(struct cudaDeviceProp));

		cudaError_t state = cudaGetDeviceProperties(prop, 0);
		if(state != cudaSuccess) { 
			//std::cerr << std::string(cudaGetErrorString(state)) <<std::endl;
			return 0;
		}
	}

	return prop;
}

