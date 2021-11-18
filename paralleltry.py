import multiprocessing as mp
import numpy as np
import time 

np.random.RandomState(100)
arr = np.random.randint(0, 10, size= [10, 5])
data = arr.tolist()
print(data)

start = time.time()

def num_range(row, min, max):
    count = 0
    for n in row:
        if min <= n <= max:
            count += 1
        return count

results = []
for row in data:
    results.append(num_range(row, 2, 5))

print(results)
end = time.time()
print(end-start)