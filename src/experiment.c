#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"

#define IDX2C(i, j, ld) ((( j ) * ( ld )) + ( i ))

const int num_values = 10;
const int values[10] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

void exp_dger() {

    for (int i = 0; i < num_values; i++) {
        int m = values[i];
	int n = values[i];

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

        free(a);
	free(x);
	free(y);
    }
}

void exp_dgemv() {

    for(int i = 0; i < num_values; i++) {
        int m = values[i];
	int n = values[i];

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
        free(a);
	free(x);
	free(y);
    }
}

void exp_daxpy() {

    for (int i = 0; i < num_values; i++) {
        int n = values[i];

	double *x = (double *)malloc(sizeof(double) * n);
        double *y = (double *)malloc(sizeof(double) * n);
        double alpha = 2.0;

        for(int i = 0; i < n; i++) {
            x[i] = i;
            y[i] = i + 3;
        }

        cblas_daxpy(n, alpha, x, 1, y, 1);

	free(x);
	free(y);
    }
}

void exp_dscal() {

    for (int i = 0; i < num_values; i++) {
        int n = values[i];

	double *x = (double *)malloc(n * sizeof(double));

	for (int j = 0; j < n; j++) x[j] = (double)j;

        double alpha = 2.0;
        cblas_dscal(n, alpha, x, 1);

	free(x);
    }
}

void exp_dgemm() {

    for (int i = 0; i < num_values; i++) {

        int n = values[i];
	int m = values[i];
	int k = values[i];

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

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, m, b, k, beta, c, m);

	free(a);
	free(b);
	free(c);

    }
}

int main(void) {

    printf("DGER\n");
    exp_dger();
    printf("DGEMV\n");
    exp_dgemv();
    printf("DAXPY\n");
    exp_daxpy();
    printf("DSCAL\n");
    exp_dscal();
    printf("DGEMM\n");
    exp_dgemm();
    return 0;
}
