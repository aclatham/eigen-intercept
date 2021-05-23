#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include "blas-intercept.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"



#define IDX2C(i ,j , ld ) ((( j )*( ld ))+( i ))
#define MEMORY_ALIGNMENT 4096
#define ALIGN_UP(x, size) ( ((size_t)x + (size - 1))&(~(size-1)) )
#define USE_GPU 1

cublasOperation_t translate(int trans) {
    if (trans == 111) return CUBLAS_OP_N;
    else if (trans == 112) return CUBLAS_OP_T;
    else return CUBLAS_OP_C;
}

void cblas_daxpy(int n, double alpha, double *X, int incX, double *Y, int incY) {
    
    struct timeval start, end;
    double load, kernel = 0;
    printf("blas-intercept: daxpy ");

    if (USE_GPU == 0) {
	printf("CPU ");
        gettimeofday(&start, NULL);
	if (!orig_daxpy) orig_daxpy = dlsym(RTLD_NEXT, "cblas_daxpy");
	gettimeofday(&end, NULL);

	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
	
	gettimeofday(&start, NULL);
	orig_daxpy(n, alpha, X, incX, Y, incY);
	gettimeofday(&end, NULL);

	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {
	printf("GPU ");

	gettimeofday(&start, NULL);
	cudaError_t cudaStat;
        cublasStatus_t stat;
        cublasHandle_t handle;

	// Create data for the device
	double *d_x, *d_y;
	cudaStat = cudaMalloc((void**)&d_x, n * sizeof(X));
        cudaStat = cudaMalloc((void**)&d_y, n * sizeof(Y));

	// Move data to device
	stat = cublasCreate(&handle);
	stat = cublasSetVector(n, sizeof(X), X, incX, d_x, incY);
	stat = cublasSetVector(n, sizeof(Y), Y, incX, d_y, incY);

	gettimeofday(&end, NULL);
	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
	gettimeofday(&start, NULL);
	// Execute on GPU and time
	stat = cublasDaxpy(handle, n, &alpha, d_x, incX, d_y, incY);

	// Get vector
	stat = cublasGetVector(n, sizeof(double), d_y, 1, Y, 1);
	gettimeofday(&end, NULL);

	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

        cudaFree(d_x);
	cudaFree(d_y);
	cublasDestroy(handle);
    }

    printf("n: %-10d load: %0.1f us\t kernel: %0.1f us\n", n, load, kernel);
}

void cblas_dgemm(int layout, int transA, int transB, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
    struct timeval start, end;
    printf("blas-intercept: dgemm ");

    double load, kernel= 0;

    if (USE_GPU == 0) {
	printf("CPU ");
	gettimeofday(&start, NULL);
        if (!orig_dgemm) orig_dgemm = dlsym(RTLD_NEXT, "cblas_dgemm");
	gettimeofday(&end, NULL);

	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
        
	gettimeofday(&start, NULL);
	orig_dgemm(layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	gettimeofday(&end, NULL);
	
	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {
        printf("GPU ");
	
	// Start timer for variables
	gettimeofday(&start, NULL);

        // Initialize variables
	double *d_a, *d_b, *d_c;
	double *a, *b, *c;
	cudaError_t cudaStat;
        cublasStatus_t stat;
        cublasHandle_t handle;

	cublasOperation_t ta = translate(transA);
       	cublasOperation_t tb = translate(transB);
        
	stat = cublasCreate(&handle);

	// End timer for variables
	gettimeofday(&end, NULL);

	printf("variables: %f ", (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec));

	// Start timer for cudaHostRegister
	gettimeofday(&start, NULL);

	// Register existing host memory ranges with cudaHostRegister
	cudaStat = cudaHostRegister(A, m * k * sizeof(double), cudaHostRegisterDefault);
	if (cudaStat != cudaSuccess) printf("cudaHostRegister A failed in DGEMM %d\n", cudaStat);
	cudaStat = cudaHostRegister(B, k * n * sizeof(double), cudaHostRegisterDefault);
	if (cudaStat != cudaSuccess) printf("cudaHostRegister B failed in DGEMM %d\n", cudaStat);
	cudaStat = cudaHostRegister(C, m * n * sizeof(double), cudaHostRegisterDefault);
	if (cudaStat != cudaSuccess) printf("cudaHostRegister C failed in DGEMM %d\n", cudaStat);

        // End timer for cudaHostRegister
	gettimeofday(&end, NULL);

	printf("cudaHostRegister: %f ", (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec));

        // Start timer for cudaGetDevicePointer
	gettimeofday(&start, NULL);

	// Get device pointer of mapped host memory
        cudaStat = cudaHostGetDevicePointer((void **) &d_a, (void **) A, 0);
	if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer A failed in DGEMM %d\n", cudaStat);
	cudaStat = cudaHostGetDevicePointer((void **) &d_b, (void **) B, 0);
	if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer B failed in DGEMM %d\n", cudaStat);
	cudaStat = cudaHostGetDevicePointer((void **) &d_c, (void **) C, 0);
	if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer C failed in DGEMM %d\n", cudaStat);

	gettimeofday(&end, NULL);

	printf("cudaHostGetDevicePointer: %f ", (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec));
	
	//load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

	// Start timer for kernel
	gettimeofday(&start, NULL);
        stat = cublasDgemm(handle, ta, tb, m, n, k, &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc);

	cudaStat = cudaDeviceSynchronize();
	if (cudaStat != cudaSuccess) printf("cudaDeviceSynchronize failed in DGEMM %d\n", cudaStat);
	//stat = cublasGetMatrix(m, n, sizeof(C), d_c, m, C, m);
        gettimeofday(&end, NULL);

	printf("kernel: %f ", (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec));

	//kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

        // Start timer for cleanup
	gettimeofday(&start, NULL);

	// Unregister memory ranges
	cudaStat = cudaHostUnregister(A);
	if (cudaStat != cudaSuccess) printf("cudaHostUnregister A failed in DGEMM %d\n", cudaStat);
	cudaStat = cudaHostUnregister(B);
	if (cudaStat != cudaSuccess) printf("cudaHostUnregister B failed in DGEMM %d\n", cudaStat);
	cudaStat = cudaHostUnregister(C);
	if (cudaStat != cudaSuccess) printf("cudaHostUnregister C failed in DGEMM %d\n", cudaStat);

	// Clean up remaining items
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cublasDestroy(handle);

	gettimeofday(&end, NULL);
	printf("Cleanup: %f ", (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec));
    }

    printf("m: %-10d n: %-10d k: %-10d load: %0.1f us\t kernel: %0.1f us\n", m, n, k, load, kernel);
}

void cblas_dgemv(int layout, int trans, int m, int n, double alpha, double *A, int lda, double *X, int incX, double beta, double *Y, int incY) {

    struct timeval start, end;
    printf("blas-intercept: dgemv ");
    double load, kernel = 0;

    if (USE_GPU == 0) {
        printf("CPU ");
	gettimeofday(&start, NULL);
	if (!orig_dgemv) orig_dgemv = dlsym(RTLD_NEXT, "cblas_dgemv");
        gettimeofday(&end, NULL);

	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

	gettimeofday(&start, NULL);
	orig_dgemv(layout, trans, m, n, alpha, A, lda, X, incX, beta, Y, incY);
	gettimeofday(&end, NULL);

	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {
        printf("GPU ");

	gettimeofday(&start, NULL);
	double *d_a, *d_x, *d_y;

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

	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

	cudaFree(d_a);
	cudaFree(d_x);
	cudaFree(d_y);
	cublasDestroy(handle);
    }

    printf("m: %-10d n: %-10d load: %0.1f us\t kernel: %0.1f us\n", m, n, load, kernel);
}

void cblas_dger(int layout, int m, int n, double alpha, double *X, int incX, double *Y, int incY, double *A, int lda)
{
    struct timeval start, end;
    double load, kernel = 0;
    printf("blas-intercept: dger ");

    if (USE_GPU == 0) {
        printf("CPU ");
	gettimeofday(&start, NULL);
	if (!orig_dger) orig_dger = dlsym(RTLD_NEXT, "cblas_dger");
	gettimeofday(&end, NULL);

	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
	
	gettimeofday(&start, NULL);
	orig_dger(layout, m, n, alpha, X, incX, Y, incY, A, lda);
	gettimeofday(&end, NULL);

	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {
        printf("GPU ");

	gettimeofday(&start, NULL);
	double *d_a, *d_x, *d_y;

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

	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
	cublasDestroy(handle);
	cudaFree(d_a);
	cudaFree(d_x);
	cudaFree(d_y);
    }

    printf("m: %-10d n: %-10d load: %0.1f us\t kernel: %0.1f us\n", m, n, load, kernel);
}

void cblas_dscal(int n, double alpha, double *X, int incX)
{
    struct timeval start, end;
    double load, kernel = 0;
    printf("blas-intercept: dscal ");

    if (USE_GPU == 0) {
        printf("CPU ");
	gettimeofday(&start, NULL);
	if (!orig_dscal) orig_dscal = dlsym(RTLD_NEXT, "cblas_dscal");
	gettimeofday(&end, NULL);
        
	load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
	
	gettimeofday(&start, NULL);
	orig_dscal(n, alpha, X, incX);
	gettimeofday(&end, NULL);
    
        kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {
        printf("GPU ");
	double *d_x;

	gettimeofday(&start, NULL);
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

	kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }

    printf("n: %-10d load: %0.1f us\t kernel: %0.1f us\n", n, load, kernel);  
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

