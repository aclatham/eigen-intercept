CC=gcc
INCLUDE=-I/usr/local/cuda/include
CFLAGS=-fPIC -shared -o
LIB=-ldl -L/usr/local/cuda/lib64 -lcudart -lcublas

all: src/blas-intercept.c src/blas-intercept-unified.c src/blas-intercept.h
	$(CC) $(INCLUDE) $(CFLAGS) lib/blas-intercept.so src/blas-intercept.c $(LIB)
	$(CC) $(INCLUDE) $(CFLAGS) lib/blas-intercept-unified.so src/blas-intercept-unified.c $(LIB)

example:
	gcc -g src/test.c -lblas -o test
	gcc -g src/experiment.c -lblas -o exp
