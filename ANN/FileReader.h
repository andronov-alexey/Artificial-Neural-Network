#ifndef FileReader_H
#define FileReader_H
#include <conio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <numeric>

#include "ANN.h"


class FileReader
{
public:
	FileReader(ANN* const);
	bool ReadInput();
private:
	void GetNextLine(istringstream&);
	ifstream f;
	ANN* const ann_ptr;
};


#endif