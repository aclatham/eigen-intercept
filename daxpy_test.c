#include <stdlib.h>
#include <stdio.h>
#include "cblas.h"

int main(int argc, char *argv[]) {

    int n = 10;

    double *x = (double *)malloc(sizeof(double) * n);
    double *y = (double *)malloc(sizeof(double) * n);
    double alpha = 2.0;

    for(int i = 0; i < n; i++) {
        x[i] = i;
	y[i] = i + 3;
    }

    printf("x: \n");
    for(int i = 0; i < n; i++) {
        printf("%f ", x[i]);
    }
    printf("\n");

    printf("y: \n");
    for(int i = 0; i < n; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");

    cblas_daxpy(n, alpha, x, 1, y, 1);

    for(int i = 0; i < n; i++) {
        printf("%f ", y[i]);
    }
    printf("\n");
    return 0;
}
