#include "FileReader.h"


FileReader::FileReader(ANN* const ann) : ann_ptr(ann)
{
}

bool FileReader::ReadInput()
{
	bool ok_status = true;
	
	f.open ("in.txt");
	if (!f.is_open())
	{
		cout << "Cannot open file \"in.txt\"";
		ok_status = false;
		return ok_status;
	}

	// start reading file
	istringstream iss;
	//string line;
	int int_val;
	double double_val;
	auto&& weights_vec = ann_ptr->weights_m;
	GetNextLine(iss); // Layers amount 
	iss >> int_val; 
	weights_vec.resize(int_val - 1);

	GetNextLine(iss); // Neurons per layer (in layer -> out layer)
	size_t row_size, col_size;
	iss >> row_size;
	for( int i = 0, iters = int_val; i < iters - 1; i++)
	{
		iss >> col_size;
		weights_vec[i] = QSMatrix<double>(row_size + 1, col_size); // + extra bias row
		row_size = col_size;
	}
	
	// initial weights (in layer -> out layer)
	for( size_t k = 0; k < weights_vec.size(); k++)
	{
		for( size_t i = 0; i < weights_vec[k].row_count(); i++)
		{
			GetNextLine(iss); 
			for( size_t j = 0; j < weights_vec[k].col_count(); j++)
			{
				iss >> weights_vec[k](i,j);
			}
		}
	}
	
	// Connections (in layer -> out layer)
	//GetNextLine(iss); 
	//iss >> int_val;
	//if (int_val == -1) // All nods are interconnected
	//{
	//	for( size_t i = 0; i < ann_ptr->neurons.size() - 1; i++)
	//	{
	//		auto && vi = ann_ptr->neurons[i];
	//		auto && v0 = vi[0].conn;
	//		v0.resize(ann_ptr->neurons[i + 1].size());
	//		iota (begin(v0), end(v0), 0);
	//		// copy
	//		for_each(next(begin(vi)), end(vi), [v0](neuron& elem){elem.conn = v0;});
	//	}
	//}
	//else
	//{
	//	iss.seekg (0, iss.beg); // set cursor to the beginning
	//	for( size_t i = 0; i < ann_ptr->neurons.size() - 1; i++)
	//	{
	//		auto && vi = ann_ptr->neurons[i];
	//		for( size_t j = 0; j < vi.size(); j++)
	//		{
	//			iss >> int_val;
	//			vi[j].conn.resize(int_val);
	//		}
	//		GetNextLine(iss);
	//		for( size_t j = 0; j < vi.size(); j++)
	//		{
	//			auto&& connects = vi[j].conn;
	//			for( size_t z = 0; z < connects.size(); z++)
	//			{
	//				iss >> connects[z];
	//			}
	//		}
	//		GetNextLine(iss);
	//	}
	//}
	
	GetNextLine(iss);
	// input signals

	ann_ptr->input.resize(weights_vec[0].row_count() - 1); // - bias row
	for( size_t i = 0; i < ann_ptr->input.size(); i++)
		iss >> ann_ptr->input[i];

	// desired output
	GetNextLine(iss);
	ann_ptr->desired_output.resize(weights_vec[weights_vec.size() - 1].row_count());
	for( size_t i = 0; i < ann_ptr->desired_output.size(); i++)
		iss >> ann_ptr->desired_output[i];
	
	f.close();
	return ok_status;
}

// Getting rid of comments
void FileReader::GetNextLine(istringstream& iss)
{
	//if (!iss.eof())
	//	return;
	string line;
	while(getline(f, line)) 
	{
		std::size_t found = line.find("//");
		if (found != string::npos)
		{
			line = string(begin(line), begin(line) + found);
		}
		if(!line.empty()) 
		{
			iss.str(line);
			iss.clear();
			break;
		}	
	}
}