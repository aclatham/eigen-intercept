#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"

#define IDX2C(i ,j , ld ) ((( j )*( ld ))+( i ))

void test_dger() {
    int m = 6;
    int n = 5;

    int pass = 1;

    double *a = (double *)malloc(m* n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    double *y = (double *)malloc(m * sizeof(double));

    int ind = 11;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            a[IDX2C(i, j, m)] = (double)ind++;
	}
    }

    for (int i = 0; i < m; i++) x[i] = 1.0;
    for (int i = 0; i < n; i++) y[i] = 1.0;

    double alpha = 2.0;

    cblas_dger(CblasColMajor, m, n, alpha, x, 1, y, 1, a, m);

    double verify[6][5] = {{13, 19, 25, 31, 37},
	                   {14, 20, 26, 32, 38},
			   {15, 21, 27, 33, 39},
			   {16, 22, 28, 34, 40},
			   {17, 23, 29, 35, 41},
			   {18, 24, 30, 36, 42}};


    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
	    if (a[IDX2C(i, j, m)] - verify[i][j] > 1e-10) pass = 0;
	}
    }

    if (pass == 1) {
        printf("dger OK\n");
    }
    else {
        printf("dger FAILED\n");
    }

    free(a);

}

void test_dgemv() {
    int m = 6;
    int n = 5;

    int pass = 1;

    double *a = (double *)malloc(m * n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    double *y = (double *)malloc(m * sizeof(double));

    int ind = 11;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            a[IDX2C(i, j, m)] = (double)ind++;
	}
    }

    for(int i = 0; i < n; i++) x[i] = 1.0;
    for(int i = 0; i < m; i++) y[i] = 0.0;

    double alpha = 1.0;
    double beta = 0.0;

    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, a, m, x, 1, beta, y, 1);

    double verify[6] = {115, 120, 125, 130, 135, 140};

    for(int j = 0; j < m; j++) {
        if (y[j] - verify[j] > 1e-10) pass = 0;
        //printf("%5.0f\n", y[j]);
    } 

    if (pass == 1) {
        printf("dgemv OK\n");
    }
    else {
        printf("dgemv FAILED\n");
    }

    free(a);
    free(x);
    free(y);
}

void test_daxpy() {
    int n = 10;
    int pass = 1;
    double *x = (double *)malloc(sizeof(double) * n);
    double *y = (double *)malloc(sizeof(double) * n);
    double alpha = 2.0;

    for(int i = 0; i < n; i++) {
        x[i] = i;
	y[i] = i + 3;
    }

    cblas_daxpy(n, alpha, x, 1, y, 1);

    double verify[10] = {3, 6, 9, 12, 15, 18, 21, 24, 27, 30};

    for(int i = 0; i < n; i++) {
	if (y[i] - verify[i] > 1e-10) pass = 0;
        //printf("%f ", y[i]);
    }
    //printf("\n");
    if (pass == 1) {
        printf("daxpy OK\n");
    }
    else {
        printf("daxpy FAILED\n");
    }

    free(x);
    free(y);

}

void test_sgemm() {

    int pass = 1;

    int n = 4;
    int m = 6;
    int k = 5;

    float *a = (float *)malloc(m * k * sizeof(float));
    float *b = (float *)malloc(k * n * sizeof(float));
    float *c = (float *)malloc(m * n * sizeof(float));

    int ind = 11;

    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            a[IDX2C(i, j, m)] = (float)ind++;
        }
    }

    ind = 11;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < k; i++) {
            b[IDX2C(i, j, k)] = (float)ind++;
        }
    }

    ind = 11;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            c[IDX2C(i, j, m)] = (float)ind++;
        }
    }

    float alpha = 1.0;
    float beta = 1.0;

    float verify[6][4] = {{1566, 2147, 2728, 3309},
                           {1632, 2238, 2844, 3450},
                           {1698, 2329, 2960, 3591},
                           {1764, 2420, 3076, 3732},
                           {1830, 2511, 3192, 3873},
                           {1896, 2602, 3308, 4014}};

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, m, b, k, beta, c, m);

    //printf ("c:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (c[IDX2C(i, j, m)] - verify[i][j] > 1e-10) {
                pass = 0;
            }
            //printf (" %5.0f",c[ IDX2C (i,j,m )]);
        }
        //printf ("\n");
    }

    if (pass == 1) {
        printf("sgemm OK\n");
    }
    else {
        printf("sgemm FAILED\n");
    }

    free(a);
    free(b);
    free(c);
}

void test_dscal() {
    int n = 6;
    double *x = (double *)malloc(n * sizeof(double));
    int pass = 1;
    for (int j = 0; j < n; j++) x[j] = (double)j;

    double alpha = 2.0;
    cblas_dscal(n, alpha, x, 1);

    double verify[6] = {0, 2, 4, 6, 8, 10};

    for (int j = 0; j < n; j++) {
        if (x[j] - verify[j] > 1e-10) pass = 0;
    }

    if (pass == 1) {
        printf("dscal OK\n");
    }
    else {
        printf("dscal FAILED\n");
    }

}


void test_dgemm() {

    int pass = 1;

    int n = 4;
    int m = 6;
    int k = 5;

    double *a = (double *)malloc(m * k * sizeof(double));
    double *b = (double *)malloc(k * n * sizeof(double));
    double *c = (double *)malloc(m * n * sizeof(double));

    int ind = 11;

    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            a[IDX2C(i, j, m)] = (double)ind++;
	}
    }

    ind = 11;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < k; i++) {
            b[IDX2C(i, j, k)] = (double)ind++;
        }
    }

    ind = 11;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            c[IDX2C(i, j, m)] = (double)ind++;
        }
    }

    double alpha = 1.0;
    double beta = 1.0;

    double verify[6][4] = {{1566, 2147, 2728, 3309},
                           {1632, 2238, 2844, 3450},
                           {1698, 2329, 2960, 3591},
                           {1764, 2420, 3076, 3732},
                           {1830, 2511, 3192, 3873},
                           {1896, 2602, 3308, 4014}};

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, m, b, k, beta, c, m);

    //printf ("c:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
	    if (c[IDX2C(i, j, m)] - verify[i][j] > 1e-10) {
                pass = 0;
	    }
            //printf (" %5.0f",c[ IDX2C (i,j,m )]);
        }
        //printf ("\n");
    }

    if (pass == 1) {
        printf("dgemm OK\n");
    }
    else {
        printf("dgemm FAILED\n");
    }

    free(a);
    free(b);
    free(c);
}



int main(void) {
    test_dger();
    test_dgemv();
    test_dgemm();
    test_daxpy();
    test_daxpy();
    test_sgemm();
    test_dscal();
    return 0;
}
