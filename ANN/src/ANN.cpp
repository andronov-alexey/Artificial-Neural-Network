#include "ANN.h"

using namespace std;

using func::BINARY;
using func::UNARY;

ANN::ANN() {
	in.ReadInput();
	InitSizes();

	maxError = 1e-5;
	gamma = 0.15;
}

void ANN::Start() {
	iteration = 0;
	FeedForward();
	for( ; errors[iteration] > maxError; ++iteration) {
		BackPropagation();
		FeedForward();
	}
}

void ANN::InitSizes() {
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
	auto F  = functions.GetFunction(UNARY::Fh);
	// derivative of the activation function
	auto dF = functions.GetFunction(UNARY::dFh);

	for (size_t i = 0; i < lastLayerIndex; i++) {
		layer_t& Oj = o[i + 1];
		layer_t& dOj = d_o[i + 1];
		// calculate excitation of a layer (aka Net(j))
		SubFill(Oj, o[i] * in.weights[i]);
		// last layer - special case
		if (i == lastLayerIndex - 1) {
			F  = functions.GetFunction(UNARY::Fo);
			dF = functions.GetFunction(UNARY::dFo);
		}

		auto arg = Oj;
		auto endIt = (i != lastLayerIndex - 1) ? prev(end(arg)) : end(arg);
		// apply function to the layer (aka s(Net(j)))
		transform(begin(arg), endIt, begin(Oj), F);
		// calculate derivatives (aka s'(Net(j)))
		transform(begin(arg), endIt, begin(dOj), dF);
	}
	// process error level
	layer_t& Olast = o[lastLayerIndex];
	layer_t& Oe =    o[errorLayerIndex];
	layer_t& dOe = d_o[errorLayerIndex];
	transform(begin(Olast), end(Olast), begin(in.desired_output), begin(Oe),  functions.GetFunction(BINARY::Fe));
	transform(begin(Olast), end(Olast), begin(in.desired_output), begin(dOe), functions.GetFunction(BINARY::dFe));

	CalculateError();
}

void ANN::CalculateError() {
	const layer_t& Oe = o[errorLayerIndex];
	errors.push_back(accumulate(begin(Oe), end(Oe), static_cast<elem_t>(0)));
}

void ANN::BackPropagation() {
	layer_t& dOlast = d_o[lastLayerIndex];
	layer_t& dOe =    d_o[errorLayerIndex];
	// backpropagated error up to the output layer
	transform(begin(dOlast), end(dOlast), begin(dOe), begin(deltas[lastLayerIndex - 1]), multiplies<elem_t>());
	// backpropagated error up to the hidden layer
	for(int i = lastLayerIndex - 2; i > -1; --i) {
		layer_t& dOprev = d_o[i + 1];
		auto WDelta = in.weights[i + 1] * deltas[i + 1];
		transform(begin(dOprev), end(dOprev), begin(WDelta), begin(deltas[i]), multiplies<elem_t>());
	}
	
	const size_t sz = deltas.size();
	// compute augmentation matrices
	for (size_t i = 0; i < sz; i++) {
		//layer_t dumbDelta (2, 10); dumbDelta[1] = 100;
		//layer_t dumbO (5, 1); dumbO[1] = 10; dumbO[2] = 100; dumbO[3] = 1000; dumbO[4] = 10000;
		//MatrixMult(accretion[0], dumbO, dumbDelta);
		//// apply learning coefficient
		//accretion[i].multscal(gamma);
		//// correct weights
		//in.weights[i] += accretion[i];
		MatrixMult(accretion[i], o[i], deltas[i]);
		// apply learning coefficient
		accretion[i].multscal(gamma);
		// correct weights
		in.weights[i] += accretion[i];
	}
}
