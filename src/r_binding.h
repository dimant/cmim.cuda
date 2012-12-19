#ifndef R_BINDING_H
#define R_BINDING_H

#if (defined _WIN32 || defined _WIN64)
	#ifdef BFSCUDA_EXPORTS
		#define BFSCUDA_DLL __declspec( dllexport )
	#else
		#define BFSCUDA_DLL __declspec( dllimport )
	#endif
#else
	#define BFSCUDA_DLL    
#endif

extern "C" {
	BFSCUDA_DLL void r_BFSCuda(char** data, int* n, int* p, int* k, int* max_parallelism, double* weights, double* inidices);
	BFSCUDA_DLL void r_mRRmatrix(char** data, int* n, int* p, int* max_parallelism, double* matrix);
	BFSCUDA_DLL void r_HardwareSupport(char** name, int* memory, int* major, int* minor, char** status);
	BFSCUDA_DLL void r_MRMR(char** data, int* n, int* p, double* indices);
}

#endif