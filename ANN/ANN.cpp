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

template <typename T> const T& 
linear(const T& val) {
	return val;
}

ANN::ANN()
{
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
	for (size_t i = 0; i < s - 1; i++) {
		m[i + 1] = m[i] * in.weights[i]; // exitation Net(j)
		// + (*sign)
	}	
}
