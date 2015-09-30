template <typename T>
bool InputStruct<T>::ReadInput()
{
	file.open ("in.txt");
	if (!file.is_open()) {
		cout << "Cannot open file \"in.txt\"";
		return false;
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
	const auto w_s = weights.size();
	if (!w_s) {
		cout << "No weight values found in input file!" << endl;
		return false;
	}
	
	for( size_t k = 0; k < w_s; k++)
	{
		const auto w_r = weights[k].row_count() - 1;
		const auto w_c = weights[k].col_count();
		for( size_t i = 0; i < w_r; i++)
		{
			for( size_t j = 0; j < w_c; j++) {
				GetNextVal(float_val); 
				weights[k](i,j) = float_val;
			}
		}
	}

	// biases
	for( size_t k = 0; k < w_s; k++)
	{
		const auto w_r = weights[k].row_count() - 1;
		const auto w_c = weights[k].col_count();
		for( size_t j = 0; j < w_c; j++) {
			GetNextVal(float_val); 
			weights[k](w_r, j) = float_val;
		}
	}

	// input signals
	input.resize(weights[0].row_count() - 1); // - bias row
	for( size_t i = 0; i < input.size(); i++) {
		GetNextVal(float_val);
		input[i] = float_val;
	}

	// desired output
	desired_output.resize(weights[weights.size() - 1].col_count());
	for( size_t i = 0; i < desired_output.size(); i++)	{
		GetNextVal(float_val);
		desired_output[i] = float_val;
	}

	file.close();
	return true;
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
