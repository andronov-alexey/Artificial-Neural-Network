#ifndef ANN_H
#define ANN_H

#include "InputStruct.h"

using namespace std;


class ANN
{
	// precision
	typedef double float_t;
public:
	ANN();
	void FeedForward();
private:
	InputStruct<float_t> in;
	vector<vector<float_t>> m;
};

#endif