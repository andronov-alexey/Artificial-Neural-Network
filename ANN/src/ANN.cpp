#include "ANN.h"

using namespace std;

//1) In case of classifier:
//hidden layer = output layer:
//	s(z) = (1 + e^-z)^-1 
//2) In case of regression:
//hidden layer:
//	s(z) = (1 + e^-z)^-1 
//output layer:
//In case of regression:
//hidden layer:
//	s(z) = (1 + e^-z)^-1 
//output layer:
//	r(z) = z //linear

template <typename T> T 
sigmoid(const T& val) {
		return 1.0 / (1.0 + exp(-val));
}

template <typename T> T 
linear(const T& val) {
	return val;
}

ANN::ANN() : iteration(0) {
	in.ReadInput();
	InitFunctions();
	InitSizes();

	maxError = 1e-5;
	gamma = 10;
}

void ANN::InitFunctions()
{
	u_functions.resize(UNARY::SIZE);
	b_functions.resize(BINARY::SIZE);

	u_functions[UNARY::Fh]  = [](const elem_t& el) { return sigmoid(el);  };
	u_functions[UNARY::dFh] = [](const elem_t& el) { return el*(1. - el); };
	b_functions[BINARY::Fe]  = [](const elem_t& o, const elem_t& d) { return 0.5*(o - d)*(o - d); };
	b_functions[BINARY::dFe] = [](const elem_t& o, const elem_t& d) { return (o - d); };
#ifdef classifier
	u_functions[UNARY::Fo]  = u_functions[UNARY::Fh];
	u_functions[UNARY::dFo] = u_functions[UNARY::dFh];
#else // regression
	u_functions[UNARY::Fo]  = [](const elem_t& el) { return linear(el); };
	u_functions[UNARY::dFo] = [](const elem_t& el) { return 0; };
#endif
}

void ANN::InitSizes()
{
	const auto s = in.weights.size();
	o.resize(s + 2); // +1 = error level, +1 = input level
	d_o.resize(s + 2);

	// fill input vector
	if (!o.empty())
		copy(begin(in.input), end(in.input), back_inserter(o[0])); 

	size_t colCount = 0;
	for (size_t i = 1; i < s + 1; i++) {
		colCount = in.weights[i - 1].col_count();
		o[i].resize(colCount);
		d_o[i].resize(colCount);
	}
	// add extended row component
	for (size_t i = 0; i < s; i++) {
		o[i].push_back(1);
	}
	// init error level size == last layer size
	colCount = o[s].size();
	o[s + 1].resize(colCount);
	d_o[s + 1].resize(colCount); 

	lastLayerIndex = o.size() - 2;
	errorLayerIndex = lastLayerIndex + 1;

	deltas.resize(lastLayerIndex);
	for( size_t i = 0; i < lastLayerIndex - 1; i++)
		deltas[i].resize(o[i + 1].size() - 1);
	deltas[lastLayerIndex - 1].resize(o[lastLayerIndex].size());
	// set same dimensions
	accretion = in.weights;
}

void ANN::FeedForward() {
	const auto s = o.size();
	// applicable activation function
	vector<func_t>::const_iterator F = begin(u_functions) + UNARY::Fh;
	// derivative of the activation function
	vector<func_t>::const_iterator dF = begin(u_functions) + UNARY::dFh;

	for (size_t i = 0; i < lastLayerIndex; i++) {
		layer_t& Oj = o[i + 1];
		layer_t& dOj = d_o[i + 1];
		// calculate excitation of a layer (aka Net(j))
		SubFill(Oj, o[i] * in.weights[i]);
		// last layer - special case
		if (i == lastLayerIndex - 1) {
			F  = begin(u_functions) + UNARY::Fo;
			dF = begin(u_functions) + UNARY::dFo;
		}
		// apply function to the layer (aka s(Net(j)))
		auto endIt = (i != lastLayerIndex - 1) ? prev(end(Oj)) : end(Oj);

		transform(begin(Oj), endIt, begin(Oj), *F);
		// calculate derivatives
		transform(begin(Oj), endIt, begin(dOj), *dF);
	}
	// process error level
	layer_t& Olast = o[lastLayerIndex];
	layer_t& Oe =    o[errorLayerIndex];
	layer_t& dOe = d_o[errorLayerIndex];
	transform(begin(Olast), end(Olast), begin(in.desired_output), begin(Oe),  b_functions[BINARY::Fe]);
	transform(begin(Olast), end(Olast), begin(in.desired_output), begin(dOe), b_functions[BINARY::dFe]);

	CalculateError();
}

void ANN::CalculateError() {
	const layer_t& Oe = o[errorLayerIndex];
	errors.push_back(accumulate(begin(Oe), end(Oe), static_cast<elem_t>(0)));
}

void ANN::BackPropagation() {
	// backpropagated error up to the output layer
	layer_t& dOlast = d_o[lastLayerIndex];
	layer_t& dOe =    d_o[errorLayerIndex];
	transform(begin(dOlast), end(dOlast), begin(dOe), begin(deltas[lastLayerIndex - 1]), multiplies<elem_t>());
	// backpropagated error up to the hidden layer
	for(int i = lastLayerIndex - 2; i > -1; --i) {
		layer_t& dOprev = d_o[i + 1];
		auto W2Delta = in.weights[i + 1] * deltas[i + 1];
		transform(begin(dOprev), end(dOprev), begin(W2Delta), begin(deltas[i]), multiplies<elem_t>());
	}
	// compute augmentation matrices
	const size_t sz = deltas.size();
	for (size_t i = 0; i < sz; i++) {
		//layer_t dumbO (2, 10); dumbO[1] = 100;
		//layer_t dumbDelta (5, 1); dumbDelta[1] = 10; dumbDelta[2] = 100; dumbDelta[3] = 1000; dumbDelta[4] = 10000;
		//MatrixMult(accretion[0], dumbDelta, dumbO);
		MatrixMult(accretion[i], o[i], deltas[i]);
		// apply learning coefficient
		accretion[i].multscal(gamma);
		// correct weights
		in.weights[i] += accretion[i];
	}
}
