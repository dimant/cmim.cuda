
#define BFSCUDA_EXPORTS
#include "bfscuda.h"

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <string>
#include <algorithm>
#include <cctype>

#include "cuda_utils.h"

// trim from start
static inline std::string &ltrim(std::string &s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
        s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
        return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
        return ltrim(rtrim(s));
}

int loadData(Data &d, RowIterator &ri) {
	int i, key, n_rows;
	std::vector<std::string> row;

	std::vector< std::map<std::string, int>* > compression;
	std::vector<int> next_key;

	n_rows = 0;
	for(ri.get_row(row); row.size() > 0; ri.get_row(row)) {
		for (i = 0; i < row.size(); i++) {
			if(n_rows == 0) {
				next_key.push_back(0);
				compression.push_back(new std::map<std::string, int>);
			}

			std::string trimmed_row = trim(row[i]);
			if(compression[i]->find(trimmed_row) == compression[i]->end()) {
				(*compression[i])[trimmed_row] = key = next_key[i];
				next_key[i] += 1;
			} else {
				key = (*compression[i])[trimmed_row];
			}
			
			d.h_data.push_back(key);
		}

		n_rows++;
	}

	for(i = 0; i < compression.size(); i++) {
		delete compression[i];
	}

	d.p = n_rows;
	d.q = (int) next_key.size();
	d.variable_widths = next_key;

	for(i = 0; i < d.variable_widths.size(); i++) {
		d.variable_widths[i] = (int) ceil(log((float)d.variable_widths[i]) / log(2.0f));
	}
	Logger& l = Logger::getInstance();
	std::stringstream output;

	output << "Observations: " << d.p << std::endl;
	l.info(output.str());
	output.clear();

	output << "Dimensions: " << d.q << std::endl;
	l.info(output.str());
	output.clear();

	output << "Bits for each dimension: " << std::endl;
	l.info(output.str());
	output.clear();

	for(i = 0; i < d.q; i++) {
		output << ceil(log((float)next_key[i]) / log(2.0f)) << " ";
	}
	output << std::endl;
	l.info(output.str());
	output.clear();

	std::sort(next_key.begin(), next_key.end(), std::greater<int>());
	int total_bits = 0;
	for(i = 0; i < 5 && i < d.variable_widths.size(); i++) {
		total_bits += (int) ceil(log((float)next_key[i]) / log(2.0f));
	}

	output << "Total bits for 5 widest dimensions: " << total_bits << std::endl;
	l.info(output.str());
	output.clear();

	CUDA_SAFE_CALL(cudaMallocPitch( (void**) &d.raw_ptr_data, &d.d_pitch, d.q * sizeof(int), d.p));

	int* h_data = thrust::raw_pointer_cast(&d.h_data[0]);

	CUDA_SAFE_CALL(cudaMemcpy2D(
		d.raw_ptr_data, d.d_pitch, 
		(const void *) h_data, d.q * sizeof(int), d.q * sizeof(int), d.p, 
		cudaMemcpyHostToDevice));

	return total_bits;
}
