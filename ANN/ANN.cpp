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
	Fh = [](float_t& el){ el = sigmoid(el);};
	Fo = Fh;
	#else // regression
	Fh = [](float_t& el){ el = sigmoid(el);};
	Fo = [](float_t& el){ el = linear(el);};
	#endif

	in.ReadInput();
	const auto s = in.weights.size(); 
	m.resize(s);
	for (size_t i = 0; i < s; i++)
		m[i].resize(in.weights[i].row_count(), 1.);
}

void ANN::FeedForward()
{
	const auto s = m.size();
	if (!s)
		return;
	copy(begin(in.input), end(in.input), begin(*m.begin())); 

	F = Fh;
	for (size_t i = 0; i < s - 1; i++) {
		vector<float_t>& Oj = m[i + 1];
		// Calculate excitation of a layer (aka Net(j))
		SubFill(Oj, m[i] * in.weights[i]);
		// last layer - special case
		if (i == s - 2)	F = Fo;
		// Apply function to the layer (aka s(Net(j)))
		for_each(begin(Oj), prev(end(Oj)), F);
	}	
}
