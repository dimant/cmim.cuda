#include <sstream>
#include <stdio.h>


#define BFSCUDA_EXPORTS
#include "bfscuda.h" 

#include "profiling_timer.h"

#include <cuda.h>

#include "utils.h"



int max_parallelism = 0;
int total_iterations = 0;

int parallel_process(Data &d, int k, int* vars, std::vector<struct result_record*> &results, bool flush) {
	static int i_parallel = 0;
	static int iter = 0;
	static thrust::host_vector< thrust::host_vector<int> > h_counts;
	static int** vars_queue;
	static double h_u_x = 0.0f;
	static bool is_initialized = false;

	struct result_record* rr;

	Logger& l = Logger::getInstance();
	std::stringstream output;

	if(is_initialized == false) {
		int i;

		i_parallel = 0;

		iter = 0;

		for(i = 0; i < max_parallelism * 2; i++) {
			h_counts.push_back(thrust::host_vector<int>(d.p * 2));
		}		
	
		vars_queue = (int**) malloc(sizeof(int*) * max_parallelism);		
	
		h_u_x = 0.0f;

		is_initialized = true;
	}

	// queue
	if((i_parallel < max_parallelism ) && flush == false)
	{
		vars_queue[i_parallel] = vars;
		i_parallel++;
	}

	if((!(i_parallel < max_parallelism ) && i_parallel > 0) || flush == true) {
		int i, j;
		double h_u_yx = 0.0f;
		double i_u_yx = 0.0f;

		// process

		// count all duplicates
		SAFE_CALL(count_duplicates(d, i_parallel, 5 + k * 2, vars_queue, h_counts))
		
		for(i = 0; i < i_parallel; i++) {

			if(vars_queue[i][0] == k + 1) {
				// update h_u_x
				h_u_x = evaluate_term(h_counts[i * 2], h_counts[i * 2 + 1], vars_queue[i], d.p);
			} else {
				// calculate h_u_yx using h_count_maps[i]
				h_u_yx = evaluate_term(h_counts[i * 2], h_counts[i * 2 + 1], vars_queue[i], d.p);

				i_u_yx = h_u_x - h_u_yx;

				if(i_u_yx == std::numeric_limits<double>::infinity() || i_u_yx < 0.0) {
					output << "Ooops " << iter << ": " << h_u_x << " - " << h_u_yx << " | ";
					for(j=0; j < vars_queue[i][0] * 2 + 1; j++) {
						output << vars_queue[i][j] << " ";
					}
					output << " | ";
					for(j = 0; j < d.p; j++) {
						output << (h_counts[i*2 +1][j] & 0x7FFFFFFF)<< " ";
					}
					output << std::endl;

					l.info(output.str());
				}
				
				rr = (struct result_record*) malloc(sizeof(struct result_record));
				rr->i_u_yx = i_u_yx;				
				for(j = 0; j < vars_queue[i][0] * 2 + 1; j++) {
					rr->vars[j] = vars_queue[i][j];
				}

				results.push_back(rr);
			}

			free(vars_queue[i]);
			vars_queue[i] = NULL;
			iter++;
		}

		i_parallel = 0;
	}

	if(flush == true) {
		h_counts.clear();
		free(vars_queue);
		is_initialized = false;
	}

	return 0;
}

struct result_record* result_record_dup(struct result_record* src) {
	int i;
	struct result_record* dst = (struct result_record*) malloc(sizeof(struct result_record));

	dst->i_u_yx = src->i_u_yx;

	for(i = 0; i < VAR_ARRAY_SIZE; i++) {
		dst->vars[i] = src->vars[i];
	}
	
	return dst;
}

int backward_search(Data d, int cond_vars, 
					  std::vector<struct result_record*> all_results, 
					  std::vector<struct result_record*>& best_results) {
	int i, j, k, l;
	struct result_record* min_result = 0;

	for(i = 0; i < d.q - 1 - cond_vars; i++) {
		min_result = 0;

		for(j = 0; j < all_results.size(); j++) {

			for(k = 0; k < best_results.size(); k++) {
				for(l = 4; l < all_results[j]->vars[0] * 2 + 1; l+=2) {
					if(all_results[j]->vars[l] == best_results[k]->vars[4]) {
						goto SKIP_LABEL1;
					}
				}
			}

			if(min_result == 0 || min_result->i_u_yx > all_results[j]->i_u_yx) {
				min_result = all_results[j];
			}
SKIP_LABEL1:;
		}

		best_results.push_back(result_record_dup(min_result));
	}

	for(i = 0; i < cond_vars; i++) {
		min_result = 0;
		for(j = 0; j < all_results.size(); j++) {
			for(k = 0; k < best_results.size(); k++) {
				if(best_results[k]->vars[4] == all_results[j]->vars[4])
					goto SKIP_LABEL2;
			}

			if(min_result == 0 || min_result->i_u_yx > all_results[j]->i_u_yx) {
				min_result = all_results[j];
			}
SKIP_LABEL2:;
		}
		best_results.push_back(result_record_dup(min_result));
	}

	return 0;
}

/*
static unsigned long inKB(unsigned long bytes)
{ return bytes/1024; }

static unsigned long inMB(unsigned long bytes)
{ return bytes/(1024*1024); }

static void printStats(unsigned long free, unsigned long total)
{
   Rprintf("^^^^ Free : %lu bytes (%lu KB) (%lu MB)\n", free, inKB(free), inMB(free));
   Rprintf("^^^^ Total: %lu bytes (%lu KB) (%lu MB)\n", total, inKB(total), inMB(total));
   Rprintf("^^^^ %f%% free, %f%% used\n", 100.0*free/(double)total, 100.0*(total - free)/(double)total);
}
*/
/*
void printFreeMem() {
	size_t free, total;
	int gpuCount, i;
	CUresult res;
	CUdevice dev;
	CUcontext ctx;

	cuInit(0);

	cuDeviceGetCount(&gpuCount);
	printf("Detected %d GPU\n",gpuCount);

	for (i=0; i<gpuCount; i++)
	{
		cuDeviceGet(&dev,i);
		cuCtxCreate(&ctx, 0, dev);
		res = cuMemGetInfo(&free, &total);
		if(res != CUDA_SUCCESS)
			printf("!!!! cuMemGetInfo failed! (status = %x)", res);
		printf("^^^^ Device: %d\n",i);
		printStats(free, total);
		cuCtxDetach(ctx);
   }
}
*/


int process(Data &d, int k, int _max_parallelism, std::vector<struct result_record*>& all_results) {
	int i, yi, n_subsets;
	LexSubset subsetIterator(k);
	
	size_t free_mem;
	size_t total_mem;

	cuMemGetInfo(&free_mem, &total_mem);

	//std::cout << "Free memory: " << free_mem << "bytes out of " << total_mem << "bytes " << std::endl;

	total_iterations = (d.q <= 16) ? (d.q - 1) * choose(d.q - 1, k) : 32000000;

	if(_max_parallelism == 0) {
		max_parallelism =  32000000 / d.p;
		if(max_parallelism > total_iterations)
			max_parallelism = total_iterations;
	}
	else
		max_parallelism = _max_parallelism;

	init_duplicate_counter(d, max_parallelism);

	std::vector<struct result_record*> results;
	

	// first element is the count excluding the first element itself e.g. [4, 0, 1, 2, 3]
	// second element is the target variable
	// remaining elements are the explanatory variables
	// order of variables is the same in the bit representation as it is
	// in the array

	int* vars;
	int subset[VAR_ARRAY_SIZE];

	for(i = 0; i < VAR_ARRAY_SIZE; i++) {
		subset[i] = 0;		
	}

	CPrecisionTimer timer;
	timer.start();

	int do_process = 0;

	n_subsets = 0;
	while(subsetIterator.nextSubset(subset, d.q - 1, k) == 0) {
		n_subsets++;

		vars = (int*) malloc(sizeof(int) * (2 * (k + 1) + 1));
		vars[0] = k+1;
		vars[1] = d.variable_widths[d.q - 1];
		vars[2] = d.q - 1;
		for(i = 0; i < k; i++) {
			vars[i * 2 + 3] = d.variable_widths[subset[i]];
			vars[i * 2 + 4] = subset[i];
		}
		
		parallel_process(d, k, vars, results, false);

		for(yi = 0; yi < d.q - 1; yi++) {
			vars = (int*) malloc(sizeof(int) * (2 * (k + 2) + 1));
			vars[0] = k + 2;
			vars[1] = d.variable_widths[d.q - 1];
			vars[2] = d.q - 1;
			vars[3] = d.variable_widths[yi];
			vars[4] = yi;

			do_process = 1;
			for(i = 0; i < k; i++) {
				vars[i * 2 + 5] = d.variable_widths[subset[i]];
				vars[i * 2 + 6] = subset[i];
				if(yi == subset[i]) {
					do_process = 0;
				}
			}

			if(do_process)
				SAFE_CALL(parallel_process(d, k, vars, results, false))
			else
				free(vars);
		}
	}

	SAFE_CALL(parallel_process(d, k, NULL, all_results, true))


	double zod = timer.stop();
	Logger& l = Logger::getInstance();
	std::stringstream output;
	output << "cmim.cuda took " << zod << "s for " << n_subsets * (d.q - 1) << " iterations." << std::endl << " That's "
		<< (zod/(n_subsets * d.q)) * 1000.0 << "ms per iteration." << std::endl;



	for(i = 0; i < results.size(); i++) {
		free(results[i]);
	}

	release_duplicate_counter();
	return 0;
}

//void fe_selection_cmim_cuda(int nb_samples, int nb_total_features, uint32_t **x, uint32_t *y, int nb_selected, int *selected) {
	//int i, j;

	//std::vector<size_t> index_map;
	//std::vector<float> b;

	//// process Data object to get results
	//// sort result indices by descending H(Y|X)
	//std::vector<float> result_vector = process(d, 3);
	//my_sort::sort(result_vector, b, index_map);

	//// mark the indices of the first nb_selected in *selected
	//for(i = 0; i < nb_selected; i++) {
	//	selected[i] = index_map[i];

	//	std::cout << selected[i] <<  "; " << result_vector[index_map[i]] << " | ";
	//}
	//std::cout << std::endl;

//}



/*
if(result != result || result < 0.0 || result == std::numeric_limits<double>::infinity()) {
int i,j;
cout << " Oooops! " << endl;

if(result != result)
cout << "narf" << endl;
if(result < 0.0)
cout << "zod" << endl;
if(result == std::numeric_limits<double>::infinity())
cout << "droz" << endl;


cout << "result: " << result << endl;

for(i = 0; i < vars[0] + 1; i++)
printf("%d ", vars[i]);
printf("\n");

printDeviceVector(sort_buffer);

for(j = 0; j < scanned_counts.size(); j++)
cout << (scanned_counts[j] & 0x7FFFFFFF) << " " << (const_ones[j+1] == 0x80000001) << " | ";
cout << endl;

cout << "n " << d->p << endl;
for(j = 0; j < 32; j++) {
cout << ((*d->h_count_map)[j] & 0x7FFFFFFF) << " ";
}
cout << endl;
cout << "Press Enter to continue." << endl;
cin.get();
}*/

