template <typename T>
bool InputStruct<T>::ReadInput()
{
	bool ok_status = true;

	file.open ("in.txt");
	if (!file.is_open()) {
		cout << "Cannot open file \"in.txt\"";
		ok_status = false;
		return ok_status;
	}

	// start reading file
	fix_t int_val = 0;
	T float_val = 0;
	GetNextVal(int_val); // Layers amount 
	weights.resize(int_val - 1);

	size_t row_size = 0, col_size = 0;
	GetNextVal(row_size); // Neurons per layer (in layer -> out layer)
	for( int i = 0, iters = int_val; i < iters - 1; i++)
	{
		GetNextVal(col_size);
		weights[i] = QSMatrix<T>(row_size + 1, col_size); // + extra bias row
		row_size = col_size;
	}

	// initial weights (in layer -> out layer)
	for( size_t k = 0; k < weights.size(); k++)
	{
		for( size_t i = 0; i < weights[k].row_count(); i++)
		{
			for( size_t j = 0; j < weights[k].col_count(); j++) {
				GetNextVal(float_val); 
				weights[k](i,j) = float_val;
			}
		}
	}

	// Connections (in layer -> out layer)
	//GetNextVal(iss); 
	//iss >> int_val;
	//if (int_val == -1) // All nods are interconnected
	//{
	//	for( size_t i = 0; i < neurons.size() - 1; i++)
	//	{
	//		auto && vi = neurons[i];
	//		auto && v0 = vi[0].conn;
	//		v0.resize(neurons[i + 1].size());
	//		iota (begin(v0), end(v0), 0);
	//		// copy
	//		for_each(next(begin(vi)), end(vi), [v0](neuron& elem){elem.conn = v0;});
	//	}
	//}
	//else
	//{
	//	iss.seekg (0, iss.beg); // set cursor to the beginning
	//	for( size_t i = 0; i < neurons.size() - 1; i++)
	//	{
	//		auto && vi = neurons[i];
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
	input.resize(weights[0].row_count() - 1); // - bias row
	for( size_t i = 0; i < input.size(); i++) {
		GetNextVal(float_val);
		input[i] = float_val;
	}

	// desired output
	desired_output.resize(weights[weights.size() - 1].row_count());
	for( size_t i = 0; i < desired_output.size(); i++)	{
		GetNextVal(float_val);
		desired_output[i] = float_val;
	}

	file.close();
	return ok_status;
}



template <typename T>
template <typename R>
void InputStruct<T>::GetNextVal(R& res)
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
