
filename = "output.txt"

# Open the output file
file = open(filename, 'r')

# Intercepted BLAS functions
intercepts = {'cblas_daxpy' : [0, 0.0],
              'cblas_dgemm' : [0, 0, 0],
              'cblas_dgemv' : [0, 0, 0],
              'cblas_dger' : [0, 0, 0],
              'cblas_dscal' : [0, 0],
              'cblas_sgemm' : [0, 0, 0]}

# Read line by line
while True:
    line = file.readline()

    # Check if line is empty
    if not line:
        break

    # Split line
    split = line.split()

    if split[0][:-1] == "blas-intercept":
        blas_id = split[1]

        # Update number of calls
        intercepts[blas_id][0] = int(intercepts[blas_id][0] + 1)
        
        if blas_id == "cblas_daxpy" or blas_id == "cblas_dscal":
            # Singe dimension calls
            if intercepts[blas_id][1] == 0:
                intercepts[blas_id][1] = int(split[3])
            else:
                intercepts[blas_id][1] = (intercepts[blas_id][1] + int(split[3])) / 2
        else:
            # Double dimension calls
            if intercepts[blas_id][1] == 0:
                intercepts[blas_id][1] = int(split[3])
                intercepts[blas_id][2] = int(split[5])
            else:
                intercepts[blas_id][1] = (intercepts[blas_id][1] + int(split[3])) / 2
                intercepts[blas_id][2] = (intercepts[blas_id][2] + int(split[5])) / 2

print(intercepts)
file.close()
