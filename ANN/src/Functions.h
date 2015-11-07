#ifndef Functions_H
#define Functions_H

#include <functional>

namespace func {

// type of activation function
#define classifier

struct UNARY {
	enum FNAME{
		Fh = 0, // activation function for hidden layer
		dFh,    // derivative of the activation function for hidden layer
		Fo,     // activation function for output layer
		dFo,    // derivative of the activation function for output layer
		SIZE
	};
};

struct BINARY {
	enum FNAME{
		Fe = 0, // activation function for error layer
		dFe,    // derivative of the activation function for error layer
		SIZE
	};
};

//1) In case of classifier:
//hidden layer = output layer:
//	s(z) = (1 + e^-z)^-1 
//2) In case of regression:
//hidden layer:
//	s(z) = (1 + e^-z)^-1 
//output layer:
//In case of regression:
//hidden layer:
//	s(z) = (1 + e^-z)^-1 
//output layer:
//	r(z) = z //linear

template <typename T> T 
sigmoid(const T & val) {
		return 1.0 / (1.0 + exp(-val));
}

template <typename T> T 
linear(const T & val) {
		return val;
}

template <typename T>
class Functions {
	typedef std::function<T(T &)> func_t;
	typedef std::function<T(T &, T &)> func2_t;
public:
	Functions();
	func_t  GetFunction(UNARY::FNAME F)
	{ return u_functions[F]; };

	func2_t GetFunction(BINARY::FNAME F)
	{ return b_functions[F]; };
private:
	func_t u_functions[UNARY::SIZE];
	func2_t b_functions[BINARY::SIZE];
};

template <typename T>
Functions<T>::Functions() {
	u_functions[UNARY::Fh]  = [](const T & el) { return sigmoid(el);  };
	u_functions[UNARY::dFh] = [](const T & el) { return el * (1. - el); };
	b_functions[BINARY::Fe]  = [](const T & o, const T & d) { return 0.5 * (o - d) * (o - d); };
	b_functions[BINARY::dFe] = [](const T & o, const T & d) { return (o - d); };
#ifdef classifier
	u_functions[UNARY::Fo]  = u_functions[UNARY::Fh];
	u_functions[UNARY::dFo] = u_functions[UNARY::dFh];
#else // regression
	u_functions[UNARY::Fo]  = [](const T & el) { return linear(el); };
	u_functions[UNARY::dFo] = [](const T & el) { return 0; };
#endif
};

#endif // #ifndef

} // namespace
