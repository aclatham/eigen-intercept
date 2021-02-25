#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

// CUDA & cuBLAS
#include <cuda_runtime.h>
#include "cublas_v2.h"

void (*orig_daxpy)(const int N, const double alpha, const double *X, const int incX, double *Y, const int incY);

// Original CUBLAS function pointers
cublasStatus_t (*orig_cublasSgemm)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *c, int ldc);

// Intercepted functions
cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
		int m, int n, int k,
		const float *alpha,
		const float *A, int lda,
		const float *B, int ldb,
		const float *beta,
		float *c, int ldc)
{
    printf("blas-intercept: cublasSgemm\n");
    
    orig_cublasSgemm = dlsym(RTLD_NEXT, "cublasSgemm");

    
    cublasStatus_t ret = orig_cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, c, ldc);
    return ret;
}


void cblas_daxpy(int n, double alpha, double *X, int incX, double *Y, int incY)
{
    // Variable Parameter
    int use_gpu = 0;

    // Timing utilities
    struct timeval start, end;

    if (use_gpu == 0) {
        // Load the original function with dlsym
	orig_daxpy = dlsym(RTLD_NEXT, "cblas_daxpy");

	// Execute the time the original function
	gettimeofday(&start, NULL);
	orig_daxpy(n, alpha, X, incX, Y, incY);
	gettimeofday(&end, NULL);
    }
    else {
        // Create cuBLAS function with intercepted parameters
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

	// Create data for the device
	double* d_x;
	double* d_y;
	cudaStat = cudaMalloc((void**)&d_x, n * sizeof(X));
        cudaStat = cudaMalloc((void**)&d_y, n * sizeof(Y));

	// Move data to device
	stat = cublasCreate(&handle);
	stat = cublasSetVector(n, sizeof(X), X, 1, d_x, 1);
	stat = cublasSetVector(n, sizeof(Y), Y, 1, d_y, 1);

	// Execute on GPU and time
	gettimeofday(&start, NULL);
	stat = cublasDaxpy(handle, n, &alpha, d_x, incX, d_y, incY);
        gettimeofday(&end, NULL);

	// Get vector
	stat = cublasGetVector(n, sizeof(double), d_y, 1, Y, 1);

	for (int i = 0; i < n; i++) {
	    printf("%f ", Y[i]);
	}
    }

    double elapsed = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec); 

    printf("blas-intercept: cblas_daxpy n: %d %f\n", n, elapsed);
}

void cblas_dgemm(int layout, int transA, int transB, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
    printf("blas-intercept: cblas_dgemm m: %d n: %d\n", m, n);
}

void cblas_dgemv(int layout, int trans, int m, int n, double alpha, double *A, int lda, double *X, int incX, double beta, double *Y, int incY) {
    printf("blas-intercept: cblas_dgemv m: %d n: %d\n", m, n);
}

void cblas_dger(int layout, int m, int n, double alpha, double *X, int incX, double *Y, int incY, double *A, int lda)
{
    printf("blas-intercept: cblas_dger m: %d n: %d\n", m, n);
}

void cblas_dscal(int n, double alpha, double *X, int incX)
{
    printf("blas-intercept: cblas_dscal n: %d\n", n);
}

void cblas_sgemm(int order, int transA, int transB, int m, int n, int k, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc)
{
    printf("blas-intercept: cblas_sgemm m: %d n: %d\n", m, n);
}
