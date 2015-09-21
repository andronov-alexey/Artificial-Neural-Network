#include "ANN.h"
#include "FileReader.h"


ANN::ANN()
{
	fr.reset(new FileReader(this));
	fr->ReadInput();
}

ANN::~ANN()
{
}