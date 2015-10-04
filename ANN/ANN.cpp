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
	#ifdef classifier
	Fh = [](float_t& el){ return sigmoid(el); };
	dFh = [](float_t& el){ return el*(1. - el); };
	Fo = Fh;
	dFo = dFh;
	#else // regression
	Fh = [](float_t& el){ return sigmoid(el); }; // test
	dFh = [](float_t& el){ return el*(1. - el); };
	Fo = [](float_t& el){ return linear(el); };
	dFo = [](float_t& el){ return 0; };
	#endif

	in.ReadInput();
	const auto s = in.weights.size();
	o.resize(s + 1); // +1 = error level 
	d_o.resize(s + 1); // +1 = error level 
	for (size_t i = 0; i < s; i++) {
		o[i].resize(in.weights[i].row_count(), 1.);
		d_o[i].resize(in.weights[i].row_count() - 1); // -1 = extended row component
	}
	// init error level
	o[s].resize(in.weights[s - 1].col_count());
	d_o[s].resize(in.weights[s - 1].col_count()); 
	// fill input vector
	if (s)
		copy(begin(in.input), end(in.input), begin(*o.begin())); 
}

void ANN::FeedForward()
{
	const auto s = o.size();
	const auto lastLayerlindex = s - 2;
	F = Fh;
	dF = dFh;

	for (size_t i = 0; i < lastLayerlindex; i++) {
		vector<float_t>& Oj = o[i + 1];
		vector<float_t>& dOj = d_o[i + 1];
		// Calculate excitation of a layer (aka Net(j))
		SubFill(Oj, o[i] * in.weights[i]);
		// last layer - special case
		if (i == lastLayerlindex - 1)	{
			F = Fo;
			dF = dFo;
		}
		// Apply function to the layer (aka s(Net(j)))
		transform(begin(Oj), prev(end(Oj)), begin(Oj), F);
		// calculate derivatives
		transform(begin(Oj), prev(end(Oj)), begin(dOj), dF);
	}
	// process error level
}
