#ifndef InputStruct_H
#define InputStruct_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>

#include "matrix.h"


template <typename T>
class InputStruct
{
	// fixed point precision
	typedef int fix_t;
public:
	bool ReadInput();
	// Weights of each layer
	std::vector<QSMatrix<T>> weights;
	std::vector<T> input;
	std::vector<T> desired_output;
private:
	template <typename R>
	void GetNextVal(R&);
	std::ifstream file; // input file
	std::istringstream buf; // buffer
};


#include "InputStruct.tpp"
#endif