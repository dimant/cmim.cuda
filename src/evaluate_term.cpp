#include "bfscuda.h"
#include <math.h>

#define FLIP_HIGH_BIT(x) (x & 0x7FFFFFFF)

double evaluate_term(thrust::host_vector<int> &unique_tuples, thrust::host_vector<int> &tuple_counts, int* vars, int n) {
	int i, j;
	int y_begin = 0;
	int y_width = vars[1];
	int mark_mask = 1;
	double result = 0.0f;

	for(i = 1; i < vars[0] *  2 + 1; i += 2) {
		mark_mask <<= vars[i];
	}
	mark_mask = mark_mask - 1;

	for(i = 0; i < unique_tuples.size(); i++) {
		if(unique_tuples[i] <= mark_mask)
			y_begin++;
		else
			break;
	}

	for(i = y_begin; FLIP_HIGH_BIT(tuple_counts[i]) > 0 && i < tuple_counts.size(); i++) {
		for(j = 0; j < y_begin; j++) {
			if(unique_tuples[j] == ((unique_tuples[i] & mark_mask) >> y_width)) {
				break;
			}
		}

		result += (float(FLIP_HIGH_BIT(tuple_counts[i])) / float(n)) 
			* log( float(FLIP_HIGH_BIT(tuple_counts[i])) / float(FLIP_HIGH_BIT(tuple_counts[j])) );
	}
	
	return -result;
}
