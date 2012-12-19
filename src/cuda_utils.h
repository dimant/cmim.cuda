#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <driver_types.h>
#include "utils.h"

#define CUDA_SAFE_CALL(x) 	{ \
	Logger& l = Logger::getInstance(); \
	std::stringstream output; \
	cudaError_t state = x; \
	if(state != cudaSuccess) { \
		output << std::string(cudaGetErrorString(state)) <<std::endl; \
		l.error(output.str()); \
		return 1; \
	} \
}

struct cudaDeviceProp* getDeviceProperties();


#endif