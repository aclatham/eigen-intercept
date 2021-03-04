

void (*orig_daxpy)(const int N, const double alpha, const double *X, const int incX, double *Y, const int incy);
void (*orig_dgemm)(const int layout, const int TransA, const int TransB, const int M, const int N, const int K, const double alpha, const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc);
void (*orig_dgemv)(const int layout, const int TransA, const int M, const int N, const double alpha, const double *A, const int lda, const double *X, const int incX, const double beta, double *Y, const int incY);
void (*orig_dger)(const int layout, const int M, const int N, const double alpha, const double *X, const int incX, const double *Y, const int incY, double *A, const int lda);
void (*orig_dscal)(const int N, const double alpha, double *X, const int incX);
void (*orig_sgemm)(const int layout, const int TransA, const int TransB, const int M, const int N, const int K, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc);
