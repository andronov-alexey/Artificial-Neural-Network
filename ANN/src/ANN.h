#ifndef ANN_H
#define ANN_H

#include "InputStruct.h"
#include <functional>
#include <iterator>

using namespace std;

// type of activation function
#define classifier // poor choice

class ANN
{
	// precision
	typedef double elem_t;
	typedef vector<elem_t> layer_t;
	// rectangular matrix
	typedef vector<layer_t> matrix_t;
	typedef function<elem_t(elem_t&)> func_t;
	typedef function<elem_t(elem_t&, elem_t&)> func2_t;
	
public:
	ANN();
	void FeedForward();
	void CalculateError();
	void BackPropagation();
private:
	void InitSizes();
	void InitFunctions();
	InputStruct<elem_t> in;
	// same type and dimensions as in.weights // very bad code :) 
	std::vector<QSMatrix<elem_t>> accretion; 
	// matrix of outputs
	matrix_t o;
	// matrix of corresponding derivatives of outputs
	matrix_t d_o;
	// functional derivatives (deltas)
	matrix_t deltas;

	size_t iteration;
	// learning rate
	elem_t gamma;
	elem_t maxError;
	// indices
	size_t lastLayerIndex;
	size_t errorLayerIndex;


	// quadratic errors of the network for every iteration
	vector<elem_t> errors;

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
	// activation function for error layer
	func2_t Fe;
	// derivative of the activation function for error layer
	func2_t dFe;
};

#endif