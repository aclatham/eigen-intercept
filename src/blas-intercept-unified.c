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

#define USE_GPU 0
#define VERBOSE 0

cublasOperation_t translate(int trans) {
    if (trans == 111) return CUBLAS_OP_N;
    else if (trans == 112) return CUBLAS_OP_T;
    else return CUBLAS_OP_C;
}

void cblas_daxpy(int n, double alpha, double *X, int incX, double *Y, int incY) {
    
    struct timeval start, end;
    double load, kernel = 0;
    if (VERBOSE) printf("blas-intercept: daxpy ");

    if (USE_GPU == 0) {
	//printf("CPU ");
        //gettimeofday(&start, NULL);
	if (!orig_daxpy) orig_daxpy = dlsym(RTLD_NEXT, "cblas_daxpy");
	//gettimeofday(&end, NULL);

	//load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
	
	//gettimeofday(&start, NULL);
	orig_daxpy(n, alpha, X, incX, Y, incY);
	//gettimeofday(&end, NULL);

	//kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {
	if (VERBOSE) printf("GPU ");

        // Initialize variables
        double *d_x, *d_y;
	cudaError_t cudaStat;
        cublasStatus_t stat;
        cublasHandle_t handle;

	stat = cublasCreate(&handle);

	// Register existing host memory ranges with cudaHostRegister
	cudaStat = cudaHostRegister(X, n * sizeof(double), cudaHostRegisterDefault);
	if (cudaStat != cudaSuccess) printf("cudaHostRegister X failed in DAXPY %d\n", cudaStat);
	cudaStat = cudaHostRegister(Y, n * sizeof(double), cudaHostRegisterDefault);
	if (cudaStat != cudaSuccess) printf("cudaHostRegister Y failed in DAXPY %d\n", cudaStat);

	// Get device pointer of mapped host memory
	cudaStat = cudaHostGetDevicePointer((void **) &d_x, (void *) X, 0);
	if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer X failed in DAXPY %d\n", cudaStat);
	cudaStat = cudaHostGetDevicePointer((void **) &d_y, (void *) Y, 0);
	if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer Y failed in DAXPY %d\n", cudaStat);

	// Run kernel
	stat = cublasDaxpy(handle, n, &alpha, d_x, incX, d_y, incY);
        cudaDeviceSynchronize();
	
	// Unregister memory ranges
	stat = cudaHostUnregister(X);
	if (cudaStat != cudaSuccess) printf("cudaHostUnregister X failed in DAXPY %d\n", cudaStat);
	stat = cudaHostUnregister(Y);
	if (cudaStat != cudaSuccess) printf("cudaHostUnregister Y failed in DAXPY %d\n", cudaStat);

	// Clean up remaining items
	cudaFree(d_x);
	cudaFree(d_y);
	cublasDestroy(handle);
    }

    if (VERBOSE) printf("n: %-10d load: %0.1f us\t kernel: %0.1f us\n", n, load, kernel);
}

void cblas_dgemm(int layout, int transA, int transB, int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc)
{
    struct timeval start, end;
    if (VERBOSE) printf("blas-intercept: dgemm ");

    double init, chr, chgdp, kernel, cleanup = 0;
    double load = 0;

    if (1 == 0) {
	//printf("CPU ");
	//gettimeofday(&start, NULL);
        if (!orig_dgemm) orig_dgemm = dlsym(RTLD_NEXT, "cblas_dgemm");
	//gettimeofday(&end, NULL);

	//load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
        
	//gettimeofday(&start, NULL);
	orig_dgemm(layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	//gettimeofday(&end, NULL);
	
	//kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {

	if (VERBOSE) {
            printf("GPU ");
	    gettimeofday(&start, NULL);
	}

	// Initialize variables
	double *d_a, *d_b, *d_c;
	double *a, *b, *c;
	cudaError_t cudaStat;
        cublasStatus_t stat;
        cublasHandle_t handle;
	stat = cublasCreate(&handle);
	cublasOperation_t ta = translate(transA);
       	cublasOperation_t tb = translate(transB);

        #if VERBOSE
	// End timer for variables
	gettimeofday(&end, NULL);
	init = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
        
        // Start timer for cudaHostRegister
        gettimeofday(&start, NULL);
        #endif

	// Register existing host memory ranges with cudaHostRegister
	cudaStat = cudaHostRegister(A, m * k * sizeof(double), cudaHostRegisterDefault);
        if (cudaStat != cudaSuccess) printf("cudaHostRegister A failed in DGEMM %d\n", cudaStat);
        cudaStat = cudaHostRegister(B, k * n * sizeof(double), cudaHostRegisterDefault);
        if (cudaStat != cudaSuccess) printf("cudaHostRegister B failed in DGEMM %d\n", cudaStat);
        cudaStat = cudaHostRegister(C, m * n * sizeof(double), cudaHostRegisterDefault);
        if (cudaStat != cudaSuccess) printf("cudaHostRegister C failed in DGEMM %d\n", cudaStat);

	#if VERBOSE
        // End timer for cudaHostRegister
        gettimeofday(&end, NULL);
        chr = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
	
	// Start timer for cudaGetDevicePointer
        gettimeofday(&start, NULL);
        #endif

        // Get device pointer of mapped host memory
        cudaStat = cudaHostGetDevicePointer((void **) &d_a, (void **) A, 0);
        if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer A failed in DGEMM %d\n", cudaStat);
        cudaStat = cudaHostGetDevicePointer((void **) &d_b, (void **) B, 0);
        if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer B failed in DGEMM %d\n", cudaStat);
        cudaStat = cudaHostGetDevicePointer((void **) &d_c, (void **) C, 0);
        if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer C failed in DGEMM %d\n", cudaStat);

        #if VERBOSE
	// End timer for cudaGetDevicePointer
        gettimeofday(&end, NULL);
        chgdp = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

        // Start timer for kernel
        gettimeofday(&start, NULL);
 	#endif

	// Run kernel
        stat = cublasDgemm(handle, ta, tb, m, n, k, &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc);
        cudaStat = cudaDeviceSynchronize();

        #if VERBOSE
	gettimeofday(&end, NULL);
        kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

        // Start timer for cleanup
        gettimeofday(&start, NULL);
 	#endif

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

	#if VERBOSE
        gettimeofday(&end, NULL);
        cleanup = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);	
        #endif
    }

    #if VERBOSE
    printf("m: %-10d n: %-10d k: %-10d init: %0.1f us\t CHR: %0.1f us CHGDP: %0.1f us kernel: %0.1f cleanup: %0.1f\n", m, n, k, init, chr, chgdp, kernel, cleanup);
    #endif
}

void cblas_dgemv(int layout, int trans, int m, int n, double alpha, double *A, int lda, double *X, int incX, double beta, double *Y, int incY) {
    #if VERBOSE
    	printf("blas-intercept: dgemv ");
    #endif

    struct timeval start, end;
    double load, kernel = 0;

    if (USE_GPU == 0) {
        //printf("CPU ");
	//gettimeofday(&start, NULL);
	if (!orig_dgemv) orig_dgemv = dlsym(RTLD_NEXT, "cblas_dgemv");
        //gettimeofday(&end, NULL);

	//load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

	//gettimeofday(&start, NULL);
	orig_dgemv(layout, trans, m, n, alpha, A, lda, X, incX, beta, Y, incY);
	//gettimeofday(&end, NULL);

	//kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {
	#if VERBOSE
            printf("GPU ");
        #endif
	 
	// Initialize variables
	double *d_a, *d_x, *d_y;
	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;
        cublasOperation_t t = translate(trans);
	stat = cublasCreate(&handle);

	// Register existing host memory ranges with cudaHostRegister
	cudaStat = cudaHostRegister(A, m * n * sizeof(double), cudaHostRegisterDefault);
	if (cudaStat != cudaSuccess) printf("cudaHostRegister A failed in DGEMV %d\n", cudaStat);
	cudaStat = cudaHostRegister(X, n * sizeof(double), cudaHostRegisterDefault);
	if (cudaStat != cudaSuccess) printf("cudaHostRegister X failed in DGEMV %d\n", cudaStat);
	cudaStat = cudaHostRegister(Y, m * sizeof(double), cudaHostRegisterDefault);
        if (cudaStat != cudaSuccess) printf("cudaHostRegister Y failed in DGEMV %d\n", cudaStat);

	// Get device pointer of mapped host memory
	cudaStat = cudaHostGetDevicePointer((void **) &d_a, (void *) A, 0);
	if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer A failed in DGEMV %d\n", cudaStat);
	cudaStat = cudaHostGetDevicePointer((void **) &d_x, (void *) X, 0);
	if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer X failed in DGEMV %d\n", cudaStat);
	cudaStat = cudaHostGetDevicePointer((void **) &d_y, (void *) Y, 0);
	if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer Y failed in DGEMV %d\n", cudaStat);
       
        // Run kernel	
	stat = cublasDgemv(handle, t, m, n, &alpha, d_a, lda, d_x, incX, &beta, d_y, incY);
        cudaDeviceSynchronize();

	// Unregister memory ranges
	cudaStat = cudaHostUnregister(A);
	if (cudaStat != cudaSuccess) printf("cudaHostUnregister A failed in DGEMV %d\n", cudaStat);
	cudaStat = cudaHostUnregister(X);
	if (cudaStat != cudaSuccess) printf("cudaHostUnregister X failed in DGEMV %d\n", cudaStat);
	cudaStat = cudaHostUnregister(Y);
	if (cudaStat != cudaSuccess) printf("cudaHostUnregister Y failed in DGEMV %d\n", cudaStat);

	// Clean up remaining items
	cudaFree(d_a);
	cudaFree(d_x);
	cudaFree(d_y);
	cublasDestroy(handle);
    }

    #if VERBOSE
        printf("m: %-10d n: %-10d load: %0.1f us\t kernel: %0.1f us\n", m, n, load, kernel);
    #endif
}

void cblas_dger(int layout, int m, int n, double alpha, double *X, int incX, double *Y, int incY, double *A, int lda)
{
    struct timeval start, end;
    double load, kernel = 0;
    if (VERBOSE) printf("blas-intercept: dger ");

    if (USE_GPU == 0) {
        //printf("CPU ");
	//gettimeofday(&start, NULL);
	if (!orig_dger) orig_dger = dlsym(RTLD_NEXT, "cblas_dger");
	//gettimeofday(&end, NULL);

	//load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
	
	//gettimeofday(&start, NULL);
	orig_dger(layout, m, n, alpha, X, incX, Y, incY, A, lda);
	//gettimeofday(&end, NULL);

	//kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {
	
        if (VERBOSE) printf("GPU ");

	// Initialize variables
	double *d_a, *d_x, *d_y;
	cudaError_t cudaStat;
        cublasStatus_t stat;
        cublasHandle_t handle;

	stat = cublasCreate(&handle);

	// Register existing host memory ranges with cudaHostRegister
	cudaStat = cudaHostRegister(A, m * n * sizeof(double), cudaHostRegisterDefault);
	if (cudaStat != cudaSuccess) printf("cudaHostRegister A failed in DGER %d\n", cudaStat);
        cudaStat = cudaHostRegister(X, m * sizeof(double), cudaHostRegisterDefault);
        if (cudaStat != cudaSuccess) printf("cudaHostRegister X failed in DGER %d\n", cudaStat);
	cudaStat = cudaHostRegister(Y, n * sizeof(double), cudaHostRegisterDefault);
        if (cudaStat != cudaSuccess) printf("cudaHostRegister Y failed in DGER %d\n", cudaStat);

        // Get device pointer of mapped host memory
 	cudaStat = cudaHostGetDevicePointer((void **) &d_a, (void *) A, 0);
	if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer A failed in DGER %d\n", cudaStat);
	cudaStat = cudaHostGetDevicePointer((void **) &d_x, (void *) X, 0);
        if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer X failed in DGER %d\n", cudaStat);
	cudaStat = cudaHostGetDevicePointer((void **) &d_y, (void *) Y, 0);
        if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer Y failed in DGER %d\n", cudaStat);

	// Run kernel
	stat = cublasDger(handle, m, n, &alpha, d_x, 1, d_y, 1, d_a, m);
        cudaDeviceSynchronize();

	// Unregister memory ranges
	cudaStat = cudaHostUnregister(A);
	if (cudaStat != cudaSuccess) printf("cudaHostUnregister A failed in DGER %d\n", cudaStat);
	cudaStat = cudaHostUnregister(X);
	if (cudaStat != cudaSuccess) printf("cudaHostUnregister X failed in DGER %d\n", cudaStat);
	cudaStat = cudaHostUnregister(Y);
	if (cudaStat != cudaSuccess) printf("cudaHostUnregister Y failed in DGER %d\n", cudaStat);

	// Clean up remaining items
	cudaFree(d_a);
	cudaFree(d_x);
	cudaFree(d_y);
	cublasDestroy(handle);
    }

    if (VERBOSE) printf("m: %-10d n: %-10d load: %0.1f us\t kernel: %0.1f us\n", m, n, load, kernel);
}

void cblas_dscal(int n, double alpha, double *X, int incX)
{
    struct timeval start, end;
    double load, kernel = 0;
    if (VERBOSE) printf("blas-intercept: dscal ");

    if (USE_GPU == 0) {
        //printf("CPU ");
	//gettimeofday(&start, NULL);
	if (!orig_dscal) orig_dscal = dlsym(RTLD_NEXT, "cblas_dscal");
	//gettimeofday(&end, NULL);
        
	//load = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
	
	//gettimeofday(&start, NULL);
	orig_dscal(n, alpha, X, incX);
	//gettimeofday(&end, NULL);
    
        //kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
    }
    else {
        if (VERBOSE) printf("GPU ");

	// Initialize variables
	double *d_x;
	cudaError_t cudaStat;
        cublasStatus_t stat;
        cublasHandle_t handle;

	stat = cublasCreate(&handle);

	// Register existing host memory ranges with cudaHostRegister
	cudaStat = cudaHostRegister(X, n * sizeof(double), cudaHostRegisterDefault);
	if (cudaStat != cudaSuccess) printf("cudaHostRegister X failed in DSCAL %d\n", cudaStat);

	// Get device pointer of mapped host memory
	cudaStat = cudaHostGetDevicePointer((void **) &d_x, (void *) X, 0);
	if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer X failed in DSCAL %d\n", cudaStat);

	// Run kernel
	stat = cublasDscal(handle, n, &alpha, d_x, 1);
        cudaDeviceSynchronize();

        // Unregister memory ranges
	stat = cudaHostUnregister(X);
	if (cudaStat != cudaSuccess) printf("cudaHostUnregister X failed in DSCAL %d\n", cudaStat);
    
	// Clean up remaining items
	cudaFree(d_x);
	cublasDestroy(handle);
    }

    if (VERBOSE) printf("n: %-10d load: %0.1f us\t kernel: %0.1f us\n", n, load, kernel);  
}


void cblas_sgemm(int layout, int transA, int transB, int m, int n, int k, float alpha, float *A, int lda, float *B, int ldb, float beta, float *C, int ldc)
{
    struct timeval start, end;
    if (VERBOSE) printf("blas-intercept: sgemm ");

    double init, chr, chgdp, kernel, cleanup = 0;
    if (1 == 0) {
        //printf("CPU ");
        if (!orig_sgemm) orig_sgemm = dlsym(RTLD_NEXT, "cblas_sgemm");
        //gettimeofday(&start, NULL);
        orig_sgemm(layout, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        //gettimeofday(&end, NULL);
    }
    else {
        #if VERBOSE
        printf("GPU ");

	// Start timer for variables
	gettimeofday(&start, NULL);
        #endif
	
	// Initialize variables
	float *d_a, *d_b, *d_c;
	float *a, *b, *c;
	cudaError_t cudaStat;
        cublasStatus_t stat;
        cublasHandle_t handle;
	stat = cublasCreate(&handle);
	cublasOperation_t ta = translate(transA);
       	cublasOperation_t tb = translate(transB);

        #if VERBOSE
	// End timer for variables
	gettimeofday(&end, NULL);
	init = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
        
        // Start timer for cudaHostRegister
        gettimeofday(&start, NULL);
        #endif

	// Register existing host memory ranges with cudaHostRegister
	cudaStat = cudaHostRegister(A, m * k * sizeof(float), cudaHostRegisterDefault);
        if (cudaStat != cudaSuccess) printf("cudaHostRegister A failed in DGEMM %d\n", cudaStat);
        cudaStat = cudaHostRegister(B, k * n * sizeof(float), cudaHostRegisterDefault);
        if (cudaStat != cudaSuccess) printf("cudaHostRegister B failed in DGEMM %d\n", cudaStat);
        cudaStat = cudaHostRegister(C, m * n * sizeof(float), cudaHostRegisterDefault);
        if (cudaStat != cudaSuccess) printf("cudaHostRegister C failed in DGEMM %d\n", cudaStat);

	#if VERBOSE
        // End timer for cudaHostRegister
        gettimeofday(&end, NULL);
        chr = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);
	
	// Start timer for cudaGetDevicePointer
        gettimeofday(&start, NULL);
        #endif

        // Get device pointer of mapped host memory
        cudaStat = cudaHostGetDevicePointer((void **) &d_a, (void **) A, 0);
        if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer A failed in DGEMM %d\n", cudaStat);
        cudaStat = cudaHostGetDevicePointer((void **) &d_b, (void **) B, 0);
        if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer B failed in DGEMM %d\n", cudaStat);
        cudaStat = cudaHostGetDevicePointer((void **) &d_c, (void **) C, 0);
        if (cudaStat != cudaSuccess) printf("cudaHostGetDevicePointer C failed in DGEMM %d\n", cudaStat);

        #if VERBOSE
	// End timer for cudaGetDevicePointer
        gettimeofday(&end, NULL);
        chgdp = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

        // Start timer for kernel
        gettimeofday(&start, NULL);
 	#endif

	// Run kernel
        stat = cublasSgemm(handle, ta, tb, m, n, k, &alpha, d_a, lda, d_b, ldb, &beta, d_c, ldc);
        cudaStat = cudaDeviceSynchronize();

        #if VERBOSE
	gettimeofday(&end, NULL);
        kernel = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);

        // Start timer for cleanup
        gettimeofday(&start, NULL);
 	#endif

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

	#if VERBOSE
        gettimeofday(&end, NULL);
        cleanup = (end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec);	
        #endif
    }

    #if VERBOSE
    printf("m: %-10d n: %-10d k: %-10d init: %0.1f us\t CHR: %0.1f us CHGDP: %0.1f us kernel: %0.1f cleanup: %0.1f\n", m, n, k, init, chr, chgdp, kernel, cleanup);
    #endif
}

