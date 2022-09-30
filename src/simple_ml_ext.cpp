#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <cstring>
#include <cstdlib>

namespace py = pybind11;

void matmul(const float *a, const float *b, float *c,
        size_t m, size_t n, size_t k) {
    /**
     * compute matrix multiply a * b = c
     * 
     * Args:
     *      a (const flost *): of size m * k
     *      b (const float *): of size k * n
     *      c (const float *): of size m * n
     */

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            c[i * n + j] = 0.0;
            for (size_t t = 0; t < k; ++t) {
                // c[i][j] += a[i][t] * b[t][j];
                c[i * n + j] += a[i * k + t] * b[t * n + j];
            }
        }
    }
}

void exp_norm(float *a, size_t m, size_t n) {
    /**
     * compute element-wise exp and row-wise normalization
     *
     * Args:
     *      a (const float *): of size m * n
     */

    float sum;
    size_t idx;

    for (size_t i = 0; i < m; ++i) {
        sum = 0.0;
        for (size_t j = 0; j < n; ++j) {
            idx = i * n + j;
            a[idx] = std::exp(a[idx]);
            sum += a[idx];
        }

        for (size_t j = 0; j < n; ++j) {
            a[i * n + j] /= sum;
        }
    }
}

void transpose(float *x, size_t m, size_t n) {
    /**
     * compute transposition for matix
     *
     * Args:
     *      a (const float *): of size m * n
     */

    float *y = (float *)malloc(m * n * sizeof(float));

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            y[j * m + i] = x[i * n + j];
        }
    }

    std::copy(y, y + m * n, x);
    free(y);
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch) {
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
    float *x_batch = (float *)std::malloc(batch * n * sizeof(float));
    float *g = (float *)std::malloc(n * k * sizeof(float));
    float *z = (float *)std::malloc(batch * k * sizeof(float));
    float scale = lr / static_cast<float>(batch);
    size_t x_batch_size = batch * n;
    size_t idx;

    for (size_t i = 0; i < m; i += batch) {
        const float *batch_start = static_cast<const float *>(X + i * n);
        std::copy(batch_start, batch_start + x_batch_size, x_batch);
        matmul(x_batch, theta, z, batch, k, n);
        exp_norm(z, batch, k);

        for (size_t j = 0; j < batch; ++j) {
            z[j * k + static_cast<size_t>(y[j + i])] -= 1.0; 
        }

        transpose(x_batch, batch, n);
        matmul(x_batch, z, g, n, k, batch);

        for (size_t j = 0; j < n; ++j) {
            for (size_t s = 0; s < k; ++s) {
                idx = j * k + s;
                theta[idx] -= scale * g[idx];
            }
        }
    }

    free(x_batch);
    free(g);
    free(z);
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
