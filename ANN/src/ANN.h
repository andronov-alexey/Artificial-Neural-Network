#ifndef ANN_H
#define ANN_H

#include <iterator>

#include "InputStruct.h"
#include "Functions.h"

class ANN
{
	// precision
	typedef double elem_t;
	typedef std::vector<elem_t> layer_t;
	// rectangular matrix
	typedef std::vector<layer_t> matrix_t;
	
public:
	ANN();
	void Start();
	void FeedForward();
	void CalculateError();
	void BackPropagation();
private:
	void InitSizes();

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

	func::Functions<elem_t> functions;
};

#endif
