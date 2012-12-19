#include "bfscuda.h"
#include "utils.h"

//#include "R_ext/Print.h"

Logger& Logger::getInstance() {
	static Logger l;

	return l;
}

Logger::Logger() {

}

void Logger::info(std::string str) {
	print("[info] " + str);
}

void Logger::warning(std::string str) {
	print("[warning] " + str);		
}

void Logger::error(std::string str) {
	print("[error] " + str);
}

void Logger::print(std::string str) {
#ifndef CMIM_PRINT
	printf("%s", str.c_str());
#endif
}

LexSubset::LexSubset(int k) {
	first = 1;
	m = 0;
	h = k;
}

int LexSubset::nextSubset(int* a, int n, int k) {
	int i;

	if(a[0] == n - k)
		return 1;

	if(first){
		first = 0;
	} else {
		if(m < n - h) {
			h = 0;
		}

		h = h + 1;
		m = a[k - h] + 1;
	}

	for(i = 0; i < h; i++) {
		a[k + i - h] = m + i;
	}

	return 0;
}


void print_binary_int(int x) {

	int i;
	int max = sizeof(int) * 8;
	Logger& l = Logger::getInstance();
	std::stringstream output;

	for(i = max - 1; i >= 0; i--) {
		output << ((x & (1 << i)) ? 1 : 0) << " ";
	}
	output << std::endl;

	l.info(output.str());
}

int factorial(int n) {
	int i, result;

	result = 1;

	for(i = 1; i <= n; i++) {
		result *= i;
	}

	return result;
}

int choose(int n, int k) {
	int result;

	result = factorial(n) / (factorial(n - k) * factorial(k));

	return result;
}