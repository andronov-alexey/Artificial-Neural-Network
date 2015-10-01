#ifndef ANN_H
#define ANN_H

#include "InputStruct.h"
#include <functional>

using namespace std;

// type of activation function
//#define classifier

class ANN
{
	// precision
	typedef double float_t;
public:
	ANN();
	void FeedForward();
private:
	InputStruct<float_t> in;
	vector<vector<float_t>> m;
	// applicable activation function
	function<void(float_t&)> F;
	// activation function for hidden layer
	function<void(float_t&)> Fh;
	// activation function for output layer
	function<void(float_t&)> Fo;
};

#endif