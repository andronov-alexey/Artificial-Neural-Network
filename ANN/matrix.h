#ifndef __QS_MATRIX_H
#define __QS_MATRIX_H

// src: https://github.com/mhallsmoore/quantcode/tree/master/data-structures-algorithms/matrix/

#include <vector>

template <typename T> class QSMatrix {
private:
	std::vector<std::vector<T>> mat;
	unsigned rows;
	unsigned cols;

public:
	QSMatrix();
	QSMatrix(unsigned _rows, unsigned _cols, const T& _initial = 0);// =0? tabs
	QSMatrix(const QSMatrix<T>& rhs);
	
	// Operator overloading, for "standard" mathematical matrix operations
	QSMatrix<T>& operator=(const QSMatrix<T>& rhs);
	
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

#include "matrix.cpp"

#endif