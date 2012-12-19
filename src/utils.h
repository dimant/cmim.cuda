#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <sstream>

// utils.cpp
int lexsub(int* a, int n, int k);
void print_binary_int(int x);
int choose(int n, int k);

class Logger {

public:
	static Logger& getInstance();

	void info(std::string str);

	void warning(std::string str);

	void error(std::string str);

private:
	Logger();
	Logger(Logger const&);
	void operator=(Logger const&);

	void print(std::string str);
};

#endif