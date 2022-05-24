CC=gcc
INCLUDE=-I/usr/local/cuda/include -I/home/austin/Projects/Masters/eigen-3.4.0 -I/home/austin/Projects/Masters/eigen-3.4.0/Eigen
CFLAGS=-fPIC -shared -o
LIB=-ldl -L/usr/local/cuda/lib64 -lcudart -lcublas

all: src/blas-intercept.c src/blas-intercept-unified.c src/blas-intercept.h src/eigen-intercept.cpp src/eigen-intercept.h
	$(CC) $(INCLUDE) $(CFLAGS) lib/blas-intercept.so src/blas-intercept.c $(LIB)
	$(CC) $(INCLUDE) $(CFLAGS) lib/blas-intercept-unified.so src/blas-intercept-unified.c $(LIB)
	g++ $(INCLUDE) $(CFLAGS) -pg lib/eigen-intercept.so src/eigen-intercept.cpp

example:
	gcc -g src/test.c -lblas -o test
	gcc -g src/experiment.c -lblas -o exp
	g++ $(INCLUDE) -pg src/eigenTest.cpp -o eigenTest
