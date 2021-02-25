all: blas-intercept.c
	gcc -I/usr/local/cuda/include -fPIC -shared -o blas-intercept.so blas-intercept.c -ldl -L/usr/local/cuda/lib64 -lcudart -lcublas

example:
	/usr/local/cuda/bin/nvcc test.c -lcublas
