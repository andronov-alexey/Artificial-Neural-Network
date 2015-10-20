#ifndef __QS_MATRIX_CPP
#define __QS_MATRIX_CPP

#include "matrix.h"

// Parameter Constructor
template <typename T>
QSMatrix<T>::QSMatrix(size_t rows, size_t cols, const T& init) {
	mat.resize(rows);
	for (size_t i = 0; i < mat.size(); i++) {
		mat[i].resize(cols, init);
	}
}

// Move Constructor
template <typename T>
QSMatrix<T>::QSMatrix(const QSMatrix<T>&& other) : mat(std::move(other.mat)) {
}

template <typename T>
QSMatrix<T>& QSMatrix<T>::operator=(QSMatrix<T>&& other) {
	mat = std::move(other.mat);
	return *this;
}

// Copy Constructor
template <typename T>
QSMatrix<T>::QSMatrix(const QSMatrix<T>& other) {
	mat = other.mat;
}

// Assignment Operator
template <typename T>
QSMatrix<T>& QSMatrix<T>::operator=(const QSMatrix<T>& other) {
	if (&other == this)
	  return *this;
	
	const auto rows = other.row_count();
	const auto cols = other.col_count();
	
	mat.resize(rows);
	
	for (size_t i = 0; i < rows; i++) {
		mat[i].resize(cols);
	}
	
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			mat[i][j] = other(i, j);
		}
	}	
	return *this;
}

// Addition of two matrices
template <typename T>
QSMatrix<T> QSMatrix<T>::operator+(const QSMatrix<T>& other) {
	const auto rows = other.row_count();
	const auto cols = other.col_count();

	QSMatrix result(rows, cols);
	
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			result(i,j) = this->mat[i][j] + other(i,j);
		}
	}
	return result;
}

// Cumulative addition of this matrix and another
template <typename T>
QSMatrix<T> & QSMatrix<T>::operator+=(const QSMatrix<T>& other) {
	const auto rows = other.row_count();
	const auto cols = other.col_count();
	assert(this->row_count() <= rows);
	assert(this->col_count() <= cols);

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			this->mat[i][j] += other(i,j); 
		}
	}
	return *this;
}

// Subtraction of this matrix and another
template <typename T>
QSMatrix<T> QSMatrix<T>::operator-(const QSMatrix<T>& other) {
	const auto rows = other.row_count();
	const auto cols = other.col_count();
	QSMatrix result(rows, cols);
	
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			result(i,j) = this->mat[i][j] - other(i,j);
		}
	}
	return result;
}

// Cumulative subtraction of this matrix and another
template <typename T>
QSMatrix<T>& QSMatrix<T>::operator-=(const QSMatrix<T>& other) {
	const auto rows = other.row_count();
	const auto cols = other.col_count();
	
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			this->mat[i][j] -= other(i,j);
		}
	}
	return *this;
}

// Left multiplication of this matrix and another
template <typename T>
QSMatrix<T> QSMatrix<T>::operator*(const QSMatrix<T>& other) {
	const auto rows = other.row_count();
	const auto cols = other.col_count();
	QSMatrix result(rows, cols);
	
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			for (size_t k = 0; k < rows; k++) {
				result(i, j) += this->mat[i][k] * rhs(k, j);
			}
		}
	}
	return result;
}

// Cumulative left multiplication of this matrix and another
template <typename T>
QSMatrix<T>& QSMatrix<T>::operator*=(const QSMatrix<T>& other) {
	QSMatrix result = (*this) * other;
	(*this) = result;
	return *this;
}

// vector * matrix 
template <typename T>
std::vector<T> operator*(const std::vector<T> & v, const QSMatrix<T> & m)
{
	const auto rows = m.row_count();
	const auto cols = m.col_count();
	const auto vsz = v.size();
	assert(vsz <= rows);

	vector<T> result(cols);

	for (size_t i = 0; i < cols; i++) {
		for (size_t j = 0; j < vsz; j++) {
			result[i] += v[j] * m(j,  i);
		}
	}
	return result;
}

// vector * matrix 
template <typename T>
std::vector<T>& operator*=(std::vector<T>& lhs, const QSMatrix<T>& rhs) {
	return (lhs = lhs * rhs);
}

// matrix * vector
template <typename T>
std::vector<T> QSMatrix<T>::operator*(const std::vector<T>& other) {
	const size_t rows = row_count();
	const size_t cols = col_count();
	std::vector<T> result(rows);
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			result[i] += this->mat[i][j] * other[j];
		}
	}
	return result;
}

template <typename T>
void MatrixMult(QSMatrix<T> & m, const std::vector<T> & lhs, const std::vector<T> & rhs) {
	const size_t rows = lhs.size();
	const size_t cols = rhs.size();
	assert(m.row_count() == rows);
	assert(m.col_count() == cols);

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			m(i, j) = lhs[i] * rhs[j];
		}
	}
}

template <typename Cont>
void SubFill(Cont& dst, const Cont& src)
{
	assert(src.size() <= dst.size());
	copy(begin(src), end(src), begin(dst));
}

// Calculate a transpose of this matrix
template <typename T>
QSMatrix<T> QSMatrix<T>::transpose() {
	const auto rows = row_count();
	const auto cols = col_count();
	QSMatrix result(rows, cols);
	
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			result(i,j) = this->mat[j][i];
		}
	}
	return result;
}

// Matrix/scalar addition
template <typename T>
QSMatrix<T> QSMatrix<T>::operator+(const T& val) {
	const auto rows = row_count();
	const auto cols = col_count();
	QSMatrix result(rows, cols);
	
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			result(i,j) = this->mat[i][j] + val;
		}
	}
	return result;
}

// Matrix/scalar subtraction
template <typename T>
QSMatrix<T> QSMatrix<T>::operator-(const T& val) {
	const auto rows = row_count();
	const auto cols = col_count();
	QSMatrix result(rows, cols);
	
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			result(i,j) = this->mat[i][j] - val;
		}
	}
	return result;
}

// Matrix/scalar multiplication
template <typename T>
QSMatrix<T> QSMatrix<T>::operator*(const T& val) {
	const auto rows = row_count();
	const auto cols = col_count();
	QSMatrix result(rows, cols);

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			result(i,j) = this->mat[i][j] * val;
		}
	}
	return result;
}

// matrix * scalar value [in place]
template <typename T>
QSMatrix<T> & QSMatrix<T>::multscal(const T& val) {
	for(auto it = begin(mat); it != end(mat); ++it)	{
		std::transform(begin(*it), end(*it), begin(*it), 
			std::bind1st(std::multiplies<T>(), val));
	}
	return *this;
}

// Matrix/scalar division
template <typename T>
QSMatrix<T> QSMatrix<T>::operator/(const T& val) {
	const auto rows = row_count();
	const auto cols = col_count();
	QSMatrix result(rows, cols);

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			result(i,j) = this->mat[i][j] / val;
		}
	}
	return result;
}

// Obtain a vector of the diagonal elements
template <typename T>
std::vector<T> QSMatrix<T>::diag_vec() const {
	const auto rows = row_count();
	std::vector<T> result(rows);
	
	for (size_t i = 0; i < rows; i++) {
		result[i] = this->mat[i][i];
	}
	return result;
}

// Access the individual elements
template <typename T>
T& QSMatrix<T>::operator()(const size_t& row, const size_t& col) {
	return this->mat[row][col];
}

// Access the individual elements (const)
template <typename T>
const T& QSMatrix<T>::operator()(const size_t& row, const size_t& col) const {
	return this->mat[row][col];
}

// Get the number of rows of the matrix
template <typename T>
size_t QSMatrix<T>::row_count() const {
	return mat.size();
}

// Get the number of columns of the matrix
template <typename T>
size_t QSMatrix<T>::col_count() const {
	if(!mat.size())
		return 0;
	return (*begin(mat)).size();
};

#endif