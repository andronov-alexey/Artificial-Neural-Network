#ifndef __QS_MATRIX_H
#define __QS_MATRIX_H

// src: https://github.com/mhallsmoore/quantcode/tree/master/data-structures-algorithms/matrix/

#include <vector>
#include <algorithm>
#include <assert.h> 

template <typename T> 
class QSMatrix {
private:
	std::vector<std::vector<T>> mat;
public:
	QSMatrix (size_t rows = 0, size_t cols = 0, const T & init = 0);
	QSMatrix (const QSMatrix<T> && other);
	QSMatrix (const QSMatrix<T> & other);
	
	// Operator overloading, for "standard" mathematical matrix operations
	QSMatrix<T> & operator = (const QSMatrix<T> & other);
	QSMatrix<T> & operator = (QSMatrix<T> && other); // move assignment

	// Matrix mathematical operations
	QSMatrix<T> operator + (const QSMatrix<T> & other);
	QSMatrix<T> & operator += (const QSMatrix<T> & other);
	QSMatrix<T> operator - (const QSMatrix<T> & other);
	QSMatrix<T> & operator -= (const QSMatrix<T> & other);
	QSMatrix<T> operator * (const QSMatrix<T> & other);
	QSMatrix<T> & operator *= (const QSMatrix<T> & other);
	QSMatrix<T> transpose();
	
	// Matrix/scalar operations
	QSMatrix<T> operator+ (const T & val);
	QSMatrix<T> operator- (const T & val);
	QSMatrix<T> operator* (const T & val);
	QSMatrix<T> operator/ (const T & val);
	QSMatrix<T> & multscal (const T & val);
	
	// Matrix/vector operations
	std::vector<T> operator * (const std::vector<T> & other);
	std::vector<T> diag_vec() const;
	
	// Access the individual elements
	T & operator() (const size_t& row, const size_t& col);
	const T & operator() (const size_t& row, const size_t& col) const;
	
	// row and column sizes
	size_t row_count() const;
	size_t col_count() const;
};

// external interfaces
template <typename T>
std::vector<T> operator * (const std::vector<T> & lhs, const QSMatrix<T> & rhs);
template <typename T>
std::vector<T> & operator *= (std::vector<T> & lhs, const QSMatrix<T> & rhs);
// vec[2] * vec[3] = mat [2x3]
template <typename T>
void MatrixMult(/*[out]*/ QSMatrix<T> & m, /*[in]*/ const std::vector<T> & lhs, /*[in]*/ const std::vector<T> & rhs);



#include "matrix.cpp"

#endif