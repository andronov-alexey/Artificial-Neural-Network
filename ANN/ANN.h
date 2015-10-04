#ifndef ANN_H
#define ANN_H

#include "InputStruct.h"
#include <functional>

using namespace std;

// type of activation function
#define classifier

class ANN
{
	// precision
	typedef double float_t;
	// rectangular matrix
	typedef vector<vector<float_t>> rect_matrix;
	typedef function<float_t(float_t&)> func_t;
public:
	ANN();
	void FeedForward();
private:
	InputStruct<float_t> in;
	// matrix of outputs
	rect_matrix o;
	// matrix of corresponding derivatives of outputs
	rect_matrix d_o;
	// applicable activation function
	func_t F;
	// derivative of the activation function
	func_t dF;
	// activation function for hidden layer
	func_t Fh;
	// derivative of the activation function for hidden layer
	func_t dFh;
	// activation function for output layer
	func_t Fo;
	// derivative of the activation function for output layer
	func_t dFo;
};

#endif