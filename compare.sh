rm -f no-intercept.txt
rm -f intercept.txt

./test >> no-intercept.txt
LD_PRELOAD=/home/pwigger/blas-intercept/lib/blas-intercept.so ./test >> intercept.txt

if cmp -s no-intercept.txt intercept.txt ; then
	echo "Output is the same"
else
	echo "Output is different"
fi
