#ifndef ANN_H
#define ANN_H

#include <vector>
using namespace std;

struct neuron
{
	neuron() : weight(1.){};
	double weight;
	vector<int> conn;
};


class ANN
{
	friend class FileReader;
public:
	ANN();
	~ANN();
private:
	unique_ptr<FileReader> fr;
	vector<vector<neuron>> neurons;
	vector<double> input;
};

#endif