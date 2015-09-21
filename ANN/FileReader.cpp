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
	GetNextLine(iss); // Layers amount 
	iss >> int_val; 
	ann_ptr->neurons.resize(int_val);

	GetNextLine(iss); // Neurons per layer (in layer -> out layer)
	for( int i = 0, iters = int_val; i < iters; i++)
	{
		iss >> int_val; 
		ann_ptr->neurons[i].resize(int_val);
	}
	
	// initial weights (in layer -> out layer)
	for( size_t i = 0; i < ann_ptr->neurons.size(); i++)
	{
		GetNextLine(iss); 
		for( size_t j = 0; j < ann_ptr->neurons[i].size(); j++)
		{
			iss >> double_val; 
			ann_ptr->neurons[i][j].weight = double_val;
		}
	}
	
	// Connections (in layer -> out layer)
	GetNextLine(iss); 
	iss >> int_val;
	if (int_val == -1) // All nods are interconnected
	{
		for( size_t i = 0; i < ann_ptr->neurons.size() - 1; i++)
		{
			auto && vi = ann_ptr->neurons[i];
			auto && v0 = vi[0].conn;
			v0.resize(ann_ptr->neurons[i + 1].size());
			iota (begin(v0), end(v0), 0);
			// copy
			for_each(next(begin(vi)), end(vi), [v0](neuron& elem){elem.conn = v0;});
		}
	}
	else
	{
		iss.seekg (0, iss.beg); // set cursor to the beginning
		for( size_t i = 0; i < ann_ptr->neurons.size() - 1; i++)
		{
			auto && vi = ann_ptr->neurons[i];
			for( size_t j = 0; j < vi.size(); j++)
			{
				iss >> int_val;
				vi[j].conn.resize(int_val);
			}
			GetNextLine(iss);
			for( size_t j = 0; j < vi.size(); j++)
			{
				auto&& connects = vi[j].conn;
				for( size_t z = 0; z < connects.size(); z++)
				{
					iss >> connects[z];
				}
			}
			GetNextLine(iss);
		}
	}
	
	// input signals
	ann_ptr->input.resize(ann_ptr->neurons[0].size());
	for( size_t i = 0; i < ann_ptr->input.size(); i++)
		iss >> ann_ptr->input[i];

	f.close();
	return ok_status;
}

// Getting rid of comments
void FileReader::GetNextLine(istringstream& iss)
{
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