#include "ANN.h"

//TODO:
// + utest
// + training sets
// + switch between activation functions
// + steps for in.txt
// + addr checkker
// bug: fail if in.txt has leading spaces 

int main ( int argc, char **argv)
{
	ANN ann;
	ann.FeedForward();
	//if (maxError >= errors[iteration])
	//	return;
	
	ann.BackPropagation();
	// ++iteration;
	volatile int itr = 5;
	itr++;
	//Output_results();
}




