#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void matmul(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    /**
     * A C++ version of the matrix multiplication code.  This should compute
     * the matrix product of A and B and store the result in C.  The matrices
     * are stored in row-major format, and the result should also be in row-major
     * format.
     *
     * Args:
     *     A (const float *): pointer to A data, of size m*n, stored in row
     *          major (C) format
     *     B (const float *): pointer to B data, of size n*k, stored in row
     *          major (C) format
     *     C (float *): pointer to C data, of size m*k, stored in row
     *          major (C) format
     *     m (size_t): number of rows in A and C
     *     n (size_t): number of columns in A and rows in B
     *     k (size_t): number of columns in B and C
     *
     * Returns:
     *     (None)
     */

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
            C[i * k + j] = 0;
            for (size_t l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int iterations = (m + batch - 1) / batch;
    for (int iter = 0; iter < iterations; iter++) {
        const float *x = &X[iter * batch * n]; // x: batch x n
        float *Z = new float[batch * k];     // Z: batch x k
        matmul(x, theta, Z, batch, n, k);
        for (int i = 0; i < batch * k; i++) Z[i] = exp(Z[i]); // element-wise exp
        for (int i = 0; i < batch; i++) {
            float sum = 0;
            for (int j = 0; j < k; j++) sum += Z[i * k + j];
            for (int j = 0; j < k; j++) Z[i * k + j] /= sum; // row-wise normalization
        }
        for (int i = 0; i < batch; i++) Z[i * k + y[iter * batch + i]] -= 1; // minus one-hot vector
        float *x_T = new float[n * batch];
        float *grad = new float[n * k];
        for (int i = 0; i < batch; i++) 
            for (int j = 0; j < n; j++) 
                x_T[j * batch + i] = x[i * n + j];
        matmul(x_T, Z, grad, n, batch, k);
        for (int i = 0; i < n * k; i++) theta[i] -= lr / batch * grad[i]; // SGD update
        delete[] Z;
        delete[] x_T;
        delete[] grad;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}