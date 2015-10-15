#include "ANN.h"

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

ANN::ANN()
{
	Fh =  [](const elem_t& el) { return sigmoid(el); };
	dFh = [](const elem_t& el) { return el*(1. - el); };
	Fe =  [](const elem_t& o, const elem_t& d){ return 0.5*(o - d)*(o - d); };
	dFe = [](const elem_t& o, const elem_t& d){ return (o - d); };
	#ifdef classifier
	Fo =  Fh;
	dFo = dFh;
	#else // regression
	Fo =  [](const elem_t& el) { return linear(el); };
	dFo = [](const elem_t& el) { return 0; };
	#endif

	in.ReadInput();

	const auto s = in.weights.size();
	o.resize(s + 2); // +1 = error level, +1 = input level
	d_o.resize(s + 2);

	// fill input vector
	if (!o.empty())
		copy(begin(in.input), end(in.input), back_inserter(o[0])); 

	size_t colCount = 0;
	for (size_t i = 1; i < s + 1; i++) {
		colCount = in.weights[i - 1].col_count();
		o[i].resize(colCount/* + ((i != s)? 1 : 0), 1.*/); // wtf is that comment
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
}

void ANN::FeedForward()
{
	const auto s = o.size();
	const auto lastLayerlindex = s - 2;
	F = Fh;
	dF = dFh;

	for (size_t i = 0; i < lastLayerlindex; i++) {
		layer_t& Oj = o[i + 1];
		layer_t& dOj = d_o[i + 1];
		// calculate excitation of a layer (aka Net(j))
		SubFill(Oj, o[i] * in.weights[i]);
		// last layer - special case
		if (i == lastLayerlindex - 1) {
			F = Fo;
			dF = dFo;
		}
		// apply function to the layer (aka s(Net(j)))
		transform(begin(Oj), (i != lastLayerlindex - 1) ? prev(end(Oj)) : end(Oj), begin(Oj), F);
		// calculate derivatives
		transform(begin(Oj), (i != lastLayerlindex - 1) ? prev(end(Oj)) : end(Oj), begin(dOj), dF);
	}
	// process error level
	layer_t& Olast = o[lastLayerlindex];
	layer_t& Oe =    o[lastLayerlindex + 1];
	layer_t& dOe = d_o[lastLayerlindex + 1];
	transform(begin(Olast), end(Olast), begin(in.desired_output), begin(Oe), Fe);
	transform(begin(Olast), end(Olast), begin(in.desired_output), begin(dOe), dFe);
}
