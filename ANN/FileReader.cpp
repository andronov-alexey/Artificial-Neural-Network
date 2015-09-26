#include "FileReader.h"


FileReader::FileReader(ANN* const ann) : ann_ptr(ann)
{
}

bool FileReader::ReadInput()
{
	bool ok_status = true;
	
	file.open ("in.txt");
	if (!file.is_open())
	{
		cout << "Cannot open file \"in.txt\"";
		ok_status = false;
		return ok_status;
	}

	// start reading file
	int int_val = 0;
	double double_val = 0;
	auto&& weights_vec = ann_ptr->weights_m;

	GetNextVal(int_val); // Layers amount 
	weights_vec.resize(int_val - 1);

	size_t row_size = 0, col_size = 0;
	GetNextVal(row_size); // Neurons per layer (in layer -> out layer)
	for( int i = 0, iters = int_val; i < iters - 1; i++)
	{
		GetNextVal(col_size);
		weights_vec[i] = QSMatrix<double>(row_size + 1, col_size); // + extra bias row
		row_size = col_size;
	}
	
	// initial weights (in layer -> out layer)
	for( size_t k = 0; k < weights_vec.size(); k++)
	{
		for( size_t i = 0; i < weights_vec[k].row_count(); i++)
		{
			for( size_t j = 0; j < weights_vec[k].col_count(); j++) {
				GetNextVal(double_val); 
				weights_vec[k](i,j) = double_val;
			}
		}
	}
	
	// Connections (in layer -> out layer)
	//GetNextVal(iss); 
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
	//		GetNextVal(iss);
	//		for( size_t j = 0; j < vi.size(); j++)
	//		{
	//			auto&& connects = vi[j].conn;
	//			for( size_t z = 0; z < connects.size(); z++)
	//			{
	//				iss >> connects[z];
	//			}
	//		}
	//		GetNextVal(iss);
	//	}
	//}
	
	//GetNextVal(iss);
	// input signals
	ann_ptr->input.resize(weights_vec[0].row_count() - 1); // - bias row
	for( size_t i = 0; i < ann_ptr->input.size(); i++) {
		GetNextVal(double_val);
		ann_ptr->input[i] = double_val;
	}

	// desired output
	ann_ptr->desired_output.resize(weights_vec[weights_vec.size() - 1].row_count());
	for( size_t i = 0; i < ann_ptr->desired_output.size(); i++)	{
		GetNextVal(double_val);
		ann_ptr->desired_output[i] = double_val;
	}
	
	file.close();
	return ok_status;
}


template <typename T>
void FileReader::GetNextVal(T& res)
{
	// check buffer first
	if (buf >> res)
		return;

	string line;
	while(getline(file, line)) 
	{
		// Getting rid of comments
		std::size_t found = line.find("//");
		if (found != string::npos)
		{
			line = string(begin(line), begin(line) + found);
		}
		if(!line.empty()) 
		{
			buf.str(line);
			buf.clear();
			buf >> res;
			break;
		}	
	}
}