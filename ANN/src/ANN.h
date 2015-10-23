#ifndef ANN_H
#define ANN_H

#include "InputStruct.h"
#include <functional>
#include <iterator>

// type of activation function
#define classifier // poor choice

struct UNARY {
	enum {
		Fh = 0, // activation function for hidden layer
		dFh,    // derivative of the activation function for hidden layer
		Fo,     // activation function for output layer
		dFo,    // derivative of the activation function for output layer
		SIZE
	};
};

struct BINARY {
	enum {
		Fe = 0, // activation function for error layer
		dFe,    // derivative of the activation function for error layer
		SIZE
	};
};

class ANN
{
	// precision
	typedef double elem_t;
	typedef std::vector<elem_t> layer_t;
	// rectangular matrix
	typedef std::vector<layer_t> matrix_t;
	typedef std::function<elem_t(elem_t&)> func_t;
	typedef std::function<elem_t(elem_t&, elem_t&)> func2_t;
	
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
	std::vector<elem_t> errors;
	std::vector<func_t> u_functions;
	std::vector<func2_t> b_functions;
};

#endif