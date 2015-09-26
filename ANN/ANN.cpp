#include "ANN.h"
#include "FileReader.h"


ANN::ANN()
{
	fr.reset(new FileReader(this));
	fr->ReadInput();
	const auto s = weights_m.size(); 
	output.resize(s);
	for (size_t i = 0; i < s; i++)
		output[i].resize(weights_m[i].row_count(), 1.);
}


void ANN::FeedForward()
{
	const auto s = output.size();
	copy(begin(input), end(input), begin(output[0]));
	for (size_t i = 0; i < s - 1; i++) {
		output[i + 1] = output[i] * weights_m[i]; // exitation Net(j)
		// + (*sign)
	}	
}

ANN::~ANN()
{
}