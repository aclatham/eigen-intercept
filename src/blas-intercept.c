#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include "blas-intercept.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define USE_GPU 1
#define PRINT_TIMES 0

cublasOperation_t translate(int trans) {
    if (trans == 111) return CUBLAS_OP_N;
    else if (trans == 112) return CUBLAS_OP_T;
    else return CUBLAS_OP_C;
}

void dsyrk_(int order, int uplo, int trans, int N, int K, double alpha, double *A, int lda, double beta, double *C, int ldc) {
    printf("blas-intercept: dsyrk\n");
}

void cblas_daxpy(int n, double alpha, double *X, int incX, double *Y, int incY) {
    
    struct timeval start, end;
    double load, kernel, start_time, end_time = 0;
    printf("blas-intercept: daxpy ");

    if (USE_GPU == 0) {
	printf("CPU ");

	// Time the function load time
        gettimeofday(&start, NULL);
	if (!orig_daxpy) orig_daxpy = dlsym(RTLD_NEXT, "cblas_daxpy");
	gettimeofday(&end, NULL);
	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

        // Record the kernel execution time	
	gettimeofday(&start, NULL);
        start_time = start.tv_sec * 1000000.0 + start.tv_usec;
	orig_daxpy(n, alpha, X, incX, Y, incY);
	gettimeofday(&end, NULL);

	end_time = end.tv_sec * 1000000.0 + end.tv_usec;
	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {
	printf("GPU ");

	// Time the function load time
	gettimeofday(&start, NULL);
	start_time = start.tv_sec * 1000000.0 + start.tv_usec;

	// Intialize CUDA variables
	cudaError_t cudaStat;
        cublasStatus_t stat;
        cublasHandle_t handle;

	// Create data for the device
	double *d_x, *d_y;
	cudaStat = cudaMalloc((void**)&d_x, n * sizeof(X));
        cudaStat = cudaMalloc((void**)&d_y, n * sizeof(Y));

	// Allocate data for the device
	stat = cublasCreate(&handle);
	stat = cublasSetVector(n, sizeof(X), X, incX, d_x, incY);
	stat = cublasSetVector(n, sizeof(Y), Y, incX, d_y, incY);
	gettimeofday(&end, NULL);

	// Record the function load time
	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

	gettimeofday(&start, NULL);
	
	// Call the cuBLAS version of the function
	stat = cublasDaxpy(handle, n, &alpha, d_x, incX, d_y, incY);
	stat = cublasGetVector(n, sizeof(double), d_y, 1, Y, 1);
	gettimeofday(&end, NULL);

        end_time = end.tv_sec * 1000000.0 + end.tv_usec;
	// Record the kernel execution time
	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

	// Free CUDA variables
        cudaFree(d_x);
	cudaFree(d_y);
	cublasDestroy(handle);
    }

    // Output message for logging
    printf("n: %-10d load: %0.1f us\t kernel: %0.1f us %0.1f %0.1f\n", n, load, kernel, start_time, end_time);
}

void cblas_dgemm(int layout, int transA, int transB, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
    struct timeval start, end;
    printf("blas-intercept: dgemm ");

    double load, kernel, start_time, end_time = 0;

    if (USE_GPU == 0) {
	printf("CPU ");

	// Time the function load time
	gettimeofday(&start, NULL);
        if (!orig_dgemm) orig_dgemm = dlsym(RTLD_NEXT, "cblas_dgemm");
	gettimeofday(&end, NULL);
	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
        
	// Record the kernel execution time
	gettimeofday(&start, NULL);
	start_time = start.tv_sec * 1000000.0 + start.tv_usec;
	orig_dgemm(layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	gettimeofday(&end, NULL);

        end_time = end.tv_sec * 1000000.0 + end.tv_usec;	
	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {
        printf("GPU ");
	gettimeofday(&start, NULL);
	double *d_a, *d_b, *d_c;
	start_time = start.tv_sec * 1000000.0 + start.tv_usec;

        cudaError_t cudaStat;
        cublasStatus_t stat;
        cublasHandle_t handle;

	cublasOperation_t ta = translate(transA);
       	cublasOperation_t tb = translate(transB);

	cudaStat = cudaMalloc((void**)&d_a, m * k * sizeof(A));
	cudaStat = cudaMalloc((void**)&d_b, k * n * sizeof(B));
	cudaStat = cudaMalloc((void**)&d_c, m * n * sizeof(C));

	stat = cublasCreate(&handle);
	stat = cublasSetMatrix(m, k, sizeof(A), A, lda, d_a, m);
	stat = cublasSetMatrix(k, n, sizeof(B), B, ldb, d_b, k);
	stat = cublasSetMatrix(m, n, sizeof(C), C, ldc, d_c, m);

	gettimeofday(&end, NULL);
	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

	gettimeofday(&start, NULL);
        stat = cublasDgemm(handle, ta, tb, m, n, k, &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc);

	stat = cublasGetMatrix(m, n, sizeof(C), d_c, m, C, m);
        gettimeofday(&end, NULL);

	end_time = end.tv_sec * 1000000.0 + end.tv_usec;
	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cublasDestroy(handle);
    }

    printf("m: %-10d n: %-10d k: %-10d load: %0.1f us\t kernel: %0.1f us %0.1f %0.1f\n", m, n, k, load, kernel, start_time, end_time);
}

void cblas_dgemv(int layout, int trans, int m, int n, double alpha, double *A, int lda, double *X, int incX, double beta, double *Y, int incY) {

    struct timeval start, end;
    printf("blas-intercept: dgemv ");
    double load, kernel, start_time, end_time = 0;

    if (USE_GPU == 0) {
        printf("CPU ");

	// Time the function load time
	gettimeofday(&start, NULL);
	if (!orig_dgemv) orig_dgemv = dlsym(RTLD_NEXT, "cblas_dgemv");
        gettimeofday(&end, NULL);
	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

	// Record the kernel execution time
	gettimeofday(&start, NULL);
	start_time = start.tv_sec * 1000000.0 + start.tv_usec;
	orig_dgemv(layout, trans, m, n, alpha, A, lda, X, incX, beta, Y, incY);
	gettimeofday(&end, NULL);

	end_time = end.tv_sec * 1000000.0 + end.tv_usec;
	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {
        printf("GPU ");

	gettimeofday(&start, NULL);
	double *d_a, *d_x, *d_y;

	start_time = start.tv_sec * 1000000.0 + start.tv_usec;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

        cublasOperation_t t = translate(trans);

	cudaStat = cudaMalloc((void**)&d_a, m * n * sizeof(A));
	cudaStat = cudaMalloc((void**)&d_x, n * sizeof(X));
	cudaStat = cudaMalloc((void**)&d_y, m * sizeof(Y));

        stat = cublasCreate(&handle);
	stat = cublasSetMatrix(m, n, sizeof(A), A, m, d_a, m);
	stat = cublasSetVector(n, sizeof(X), X, 1, d_x, 1);
	stat = cublasSetVector(m, sizeof(Y), Y, 1, d_y, 1);
        gettimeofday(&end, NULL);

	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

        gettimeofday(&start, NULL);
	stat = cublasDgemv(handle, t, m, n, &alpha, d_a, m, d_x, 1, &beta, d_y, 1);

	stat = cublasGetVector(m, sizeof(Y), d_y, 1, Y, 1);
        gettimeofday(&end, NULL);

	end_time = end.tv_sec * 1000000.0 + end.tv_usec;
	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

	cudaFree(d_a);
	cudaFree(d_x);
	cudaFree(d_y);
	cublasDestroy(handle);
    }

    printf("m: %-10d n: %-10d load: %0.1f us\t kernel: %0.1f us %0.1f %0.1f\n", m, n, load, kernel, start_time, end_time);
}

void cblas_dger(int layout, int m, int n, double alpha, double *X, int incX, double *Y, int incY, double *A, int lda)
{
    struct timeval start, end;
    double load, kernel, start_time, end_time = 0;
    printf("blas-intercept: dger ");

    if (USE_GPU == 0) {
        printf("CPU ");

	// Time the function load time
	gettimeofday(&start, NULL);
	if (!orig_dger) orig_dger = dlsym(RTLD_NEXT, "cblas_dger");
	gettimeofday(&end, NULL);
	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
	
	// Record the kernel execution time
	gettimeofday(&start, NULL);
	start_time = start.tv_sec * 1000000.0 + start.tv_usec;
	orig_dger(layout, m, n, alpha, X, incX, Y, incY, A, lda);
	gettimeofday(&end, NULL);

	end_time = end.tv_sec * 1000000.0 + end.tv_usec;
	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {
        printf("GPU ");

	gettimeofday(&start, NULL);
	double *d_a, *d_x, *d_y;

	start_time = start.tv_sec * 1000000.0 + start.tv_usec;
	cudaError_t cudaStat;
        cublasStatus_t stat;
        cublasHandle_t handle;

	cudaStat = cudaMalloc((void**)&d_a, m * n * sizeof(A));
        cudaStat = cudaMalloc((void**)&d_x, m * sizeof(X));
        cudaStat = cudaMalloc((void**)&d_y, n * sizeof(Y));

	stat = cublasCreate(&handle);
        stat = cublasSetMatrix(m, n, sizeof(A), A, m, d_a, m);
        stat = cublasSetVector(m, sizeof(X), X, 1, d_x, 1);
        stat = cublasSetVector(n, sizeof(Y), Y, 1, d_y, 1);
        gettimeofday(&end, NULL);

	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

	gettimeofday(&start, NULL);
	stat = cublasDger(handle, m, n, &alpha, d_x, 1, d_y, 1, d_a, m);

	stat = cublasGetMatrix(m, n, sizeof(A), d_a, m, A, m);
        gettimeofday(&end, NULL);
	end_time = end.tv_sec * 1000000.0 + end.tv_usec;

	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
	cublasDestroy(handle);
	cudaFree(d_a);
	cudaFree(d_x);
	cudaFree(d_y);
    }

    printf("m: %-10d n: %-10d load: %0.1f us\t kernel: %0.1f us %0.1f %0.1f\n", m, n, load, kernel, start_time, end_time);
}

void cblas_dscal(int n, double alpha, double *X, int incX)
{
    struct timeval start, end;
    double load, kernel, start_time, end_time = 0;
    printf("blas-intercept: dscal ");

    if (USE_GPU == 0) {
        printf("CPU ");

	// Time the function load time
	gettimeofday(&start, NULL);
	if (!orig_dscal) orig_dscal = dlsym(RTLD_NEXT, "cblas_dscal");
	gettimeofday(&end, NULL);
	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
	
	// Record the kernel execution time
	gettimeofday(&start, NULL);
	start_time = start.tv_sec * 1000000.0 + start.tv_usec;
	orig_dscal(n, alpha, X, incX);
	gettimeofday(&end, NULL);
    
	end_time = end.tv_sec * 1000000.0 + end.tv_usec;
        kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {
        printf("GPU ");
	double *d_x;

	gettimeofday(&start, NULL);
	start_time = start.tv_sec * 1000000.0 + start.tv_usec;
	cudaError_t cudaStat;
        cublasStatus_t stat;
        cublasHandle_t handle;

	stat = cudaMalloc((void**)&d_x, n * sizeof(X));

	stat = cublasCreate(&handle);
	stat = cublasSetVector(n, sizeof(X), X, 1, d_x, 1);
        gettimeofday(&end, NULL);

	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

	gettimeofday(&start, NULL);

	stat = cublasDscal(handle, n, &alpha, d_x, 1);

	stat = cublasGetVector(n, sizeof(double), d_x, 1, X, 1);
	gettimeofday(&end, NULL);

	end_time = end.tv_sec * 1000000.0 + end.tv_usec;
	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }

    printf("n: %-10d load: %0.1f us\t kernel: %0.1f us %0.1f %0.1f\n", n, load, kernel, start_time, end_time);  
}


void cblas_sgemm(int layout, int transA, int transB, int m, int n, int k, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc)
{
    struct timeval start, end;
    printf("blas-intercept: sgemm ");

    if (USE_GPU == 0) {
        printf("CPU ");

        if (!orig_sgemm) orig_sgemm = dlsym(RTLD_NEXT, "cblas_sgemm");
        gettimeofday(&start, NULL);
        orig_sgemm(layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        gettimeofday(&end, NULL);
    }
    else {
        printf("GPU ");
        float *d_a, *d_b, *d_c;

        cudaError_t cudaStat;
        cublasStatus_t stat;
        cublasHandle_t handle;

	gettimeofday(&start, NULL);
        cublasOperation_t ta = translate(transA);
        cublasOperation_t tb = translate(transB);

        cudaStat = cudaMalloc((void**)&d_a, m * k * sizeof(A));
        cudaStat = cudaMalloc((void**)&d_b, k * n * sizeof(B));
        cudaStat = cudaMalloc((void**)&d_c, m * n * sizeof(C));

        stat = cublasCreate(&handle);
        stat = cublasSetMatrix(m, k, sizeof(A), A, lda, d_a, m);
        stat = cublasSetMatrix(k, n, sizeof(B), B, ldb, d_b, k);
        stat = cublasSetMatrix(m, n, sizeof(C), C, ldc, d_c, m);

        stat = cublasSgemm(handle, ta, tb, m, n, k, &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc);

        stat = cublasGetMatrix(m, n, sizeof(C), d_c, m, C, m);
        gettimeofday(&end, NULL);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cublasDestroy(handle);
    }
    double elapsed = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

    printf("m: %d n: %d k: %d %0.1f us\n", m, n, k, elapsed);
}

