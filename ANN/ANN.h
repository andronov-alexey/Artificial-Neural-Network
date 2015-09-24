#ifndef ANN_H
#define ANN_H

#include <vector>
#include "matrix.h"

using namespace std;


class ANN
{
	friend class FileReader;
public:
	ANN();
	~ANN();
private:
	unique_ptr<FileReader> fr;
	// Weights of each layer
	vector<QSMatrix<double>> weights_m;
	vector<double> input;
	vector<double> desired_output;
};

#endif