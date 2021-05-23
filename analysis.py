

filename = "output.txt"

# Open the output file
file = open(filename, 'r')

calls = []

# Read line by line
while True:
    line = file.readline()

    # Check if line is empty
    if not line:
        break

    # Split line
    split = line.split()

    if split[0][:-1] == "blas-intercept" and split[1] != "sgemm":
        calls.append(split)


calls.sort(key = lambda calls: calls[-2])

last_start = 0
last_end = 0

for item in calls:

    overlap = False
    if float(item[-2]) < last_end:
        overlap = True


    print(item[1], end=' ')
    print(item[-2], end = ' ')
    print(item[-1], end = ' ')

    if overlap:
        print("Overlap")
    else:
        print()

    last_start = float(item[-2])
    last_end = float(item[-1])

file.close()
