#ifndef __QS_MATRIX_H
#define __QS_MATRIX_H

// src: https://github.com/mhallsmoore/quantcode/tree/master/data-structures-algorithms/matrix/

#include <vector>
#include <assert.h> 

// + swap?

template <typename T> 
class QSMatrix {
private:
	std::vector<std::vector<T>> mat;
	unsigned rows; // get rid?
	unsigned cols;

public:
	QSMatrix();
	QSMatrix(unsigned _rows, unsigned _cols, const T& _initial = 0);
	QSMatrix(const QSMatrix<T>&& other);
	QSMatrix(const QSMatrix<T>& other);
	
	// Operator overloading, for "standard" mathematical matrix operations
	QSMatrix<T>& operator=(const QSMatrix<T>& other);
	QSMatrix<T>& operator=(QSMatrix<T>&& other); // move assignment

	// Matrix mathematical operations
	QSMatrix<T> operator+(const QSMatrix<T>& rhs);
	QSMatrix<T>& operator+=(const QSMatrix<T>& rhs);
	QSMatrix<T> operator-(const QSMatrix<T>& rhs);
	QSMatrix<T>& operator-=(const QSMatrix<T>& rhs);
	QSMatrix<T> operator*(const QSMatrix<T>& rhs);
	QSMatrix<T>& operator*=(const QSMatrix<T>& rhs);
	QSMatrix<T> transpose();
	
	// Matrix/scalar operations
	QSMatrix<T> operator+(const T& rhs);
	QSMatrix<T> operator-(const T& rhs);
	QSMatrix<T> operator*(const T& rhs);
	QSMatrix<T> operator/(const T& rhs);
	QSMatrix<T>& multscal(const T& val);
	
	// Matrix/vector operations
	std::vector<T> operator*(const std::vector<T>& rhs);
	std::vector<T> diag_vec();
	
	

	// Access the individual elements
	T& operator()(const unsigned& row, const unsigned& col);
	const T& operator()(const unsigned& row, const unsigned& col) const;
	
	// Access the row and column sizes
	unsigned row_count() const;
	unsigned col_count() const;
	
};

// external interfaces
template <typename T>
std::vector<T> operator*(const std::vector<T>& lhs, const QSMatrix<T>& rhs);
template <typename T>
std::vector<T>& operator*=(std::vector<T>& lhs, const QSMatrix<T>& rhs);



#include "matrix.cpp"

#endif