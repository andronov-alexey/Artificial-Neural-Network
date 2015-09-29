#ifndef __QS_MATRIX_CPP
#define __QS_MATRIX_CPP

#include "matrix.h"

// Default Constructor
template <typename T>
QSMatrix<T>::QSMatrix() {
	rows = (cols = 0);
}

// Parameter Constructor
template <typename T>
QSMatrix<T>::QSMatrix(unsigned _rows, unsigned _cols, const T& _initial) {
	mat.resize(_rows);
	for (unsigned i = 0; i < mat.size(); i++) {
		mat[i].resize(_cols, _initial);
	}
	rows = _rows;
	cols = _cols;
}

// Copy Constructor
template <typename T>
QSMatrix<T>::QSMatrix(const QSMatrix<T>& rhs) {
	mat = rhs.mat;
	rows = rhs.row_count();
	cols = rhs.col_count();
}

// Assignment Operator
template <typename T>
QSMatrix<T>& QSMatrix<T>::operator=(const QSMatrix<T>& rhs) {
	if (&rhs == this)
	  return *this;
	
	rows = rhs.row_count();
	cols = rhs.col_count();
	
	mat.resize(rows);
	
	for (unsigned i = 0; i < mat.size(); i++) {
		mat[i].resize(cols);
	}
	
	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			mat[i][j] = rhs(i, j);
		}
	}
	 	
	return *this;
}

// Addition of two matrices
template <typename T>
QSMatrix<T> QSMatrix<T>::operator+(const QSMatrix<T>& rhs) {
	QSMatrix result(rows, cols, 0.0);
	
	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result(i,j) = this->mat[i][j] + rhs(i,j);
		}
	}
	
	return result;
}

// Cumulative addition of this matrix and another
template <typename T>
QSMatrix<T>& QSMatrix<T>::operator+=(const QSMatrix<T>& rhs) {
	unsigned rows = rhs.row_count();
	unsigned cols = rhs.col_count();
	
	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			this->mat[i][j] += rhs(i,j); 
		}
	}
	
	return *this;
}

// Subtraction of this matrix and another
template <typename T>
QSMatrix<T> QSMatrix<T>::operator-(const QSMatrix<T>& rhs) {
	unsigned rows = rhs.row_count();
	unsigned cols = rhs.col_count();
	QSMatrix result(rows, cols, 0.0);
	
	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result(i,j) = this->mat[i][j] - rhs(i,j);
		}
	}
	
	return result;
}

// Cumulative subtraction of this matrix and another
template <typename T>
QSMatrix<T>& QSMatrix<T>::operator-=(const QSMatrix<T>& rhs) {
	unsigned rows = rhs.row_count();
	unsigned cols = rhs.col_count();
	
	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			this->mat[i][j] -= rhs(i,j);
		}
	}
	
	return *this;
}

// Left multiplcation of this matrix and another
template <typename T>
QSMatrix<T> QSMatrix<T>::operator*(const QSMatrix<T>& rhs) {
	unsigned rows = rhs.row_count();
	unsigned cols = rhs.col_count();
	QSMatrix result(rows, cols, 0.0);
	
	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			for (unsigned k = 0; k < rows; k++) {
				result(i, j) += this->mat[i][k] * rhs(k, j);
			}
		}
	}
	
	return result;
}

// Cumulative left multiplication of this matrix and another
template <typename T>
QSMatrix<T>& QSMatrix<T>::operator*=(const QSMatrix<T>& rhs) {
	QSMatrix result = (*this) * rhs;
	(*this) = result;
	return *this;
}

// vector * matrix 
template <typename T>
std::vector<T> operator*(const std::vector<T>& lhs, const QSMatrix<T>& rhs)
{
	unsigned rows = rhs.row_count();
	unsigned cols = rhs.col_count();
	assert(lhs.size() == rows);

	vector<T> result(cols);

	for (unsigned i = 0; i < cols; i++) {
		for (unsigned j = 0; j < rows; j++) {
			result[i] += lhs[j] * rhs(j,  i);
		}
	}

	return result;
}

// vector * matrix 
template <typename T>
std::vector<T>& operator*=(std::vector<T>& lhs, const QSMatrix<T>& rhs) {
	return (lhs = lhs * rhs);
}



// Calculate a transpose of this matrix
template <typename T>
QSMatrix<T> QSMatrix<T>::transpose() {
	QSMatrix result(rows, cols, 0.0);
	
	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result(i,j) = this->mat[j][i];
		}
	}
	
	return result;
}

// Matrix/scalar addition
template <typename T>
QSMatrix<T> QSMatrix<T>::operator+(const T& rhs) {
	QSMatrix result(rows, cols, 0.0);
	
	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result(i,j) = this->mat[i][j] + rhs;
		}
	}
	
	return result;
}

// Matrix/scalar subtraction
template <typename T>
QSMatrix<T> QSMatrix<T>::operator-(const T& rhs) {
	QSMatrix result(rows, cols, 0.0);
	
	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			result(i,j) = this->mat[i][j] - rhs;
		}
	}
	
	return result;
}

// Matrix/scalar multiplication
template <typename T>
QSMatrix<T> QSMatrix<T>::operator*(const T& rhs) {
  QSMatrix result(rows, cols, 0.0);

  for (unsigned i = 0; i < rows; i++) {
	for (unsigned j = 0; j < cols; j++) {
		result(i,j) = this->mat[i][j] * rhs;
	}
  }

  return result;
}

// Matrix/scalar division
template <typename T>
QSMatrix<T> QSMatrix<T>::operator/(const T& rhs) {
  QSMatrix result(rows, cols, 0.0);
  
  for (unsigned i = 0; i < rows; i++) {
	for (unsigned j = 0; j < cols; j++) {
		result(i,j) = this->mat[i][j] / rhs;
	}
  }
  
  return result;
}

// Multiply a matrix with a vector
template <typename T>
std::vector<T> QSMatrix<T>::operator*(const std::vector<T>& rhs) {
  std::vector<T> result(rhs.size(), 0.0);

  for (unsigned i = 0; i < rows; i++) {
	for (unsigned j = 0; j < cols; j++) {
		result[i] = this->mat[i][j] * rhs[j];
	}
  }

  return result;
}

// Obtain a vector of the diagonal elements
template <typename T>
std::vector<T> QSMatrix<T>::diag_vec() {
	std::vector<T> result(rows);
	
	for (unsigned i = 0; i < rows; i++) {
		result[i] = this->mat[i][i];
	}
	
	return result;
}

// Access the individual elements
template <typename T>
T& QSMatrix<T>::operator()(const unsigned& row, const unsigned& col) {
	return this->mat[row][col];
}

// Access the individual elements (const)
template <typename T>
const T& QSMatrix<T>::operator()(const unsigned& row, const unsigned& col) const {
	return this->mat[row][col];
}

// Get the number of rows of the matrix
template <typename T>
unsigned QSMatrix<T>::row_count() const {
	return this->rows;
}

// Get the number of columns of the matrix
template <typename T>
unsigned QSMatrix<T>::col_count() const {
	return this->cols;
};

#endif