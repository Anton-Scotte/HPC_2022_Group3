# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:28:01 2022

@author: victo
"""


from timeit import default_timer as timer
import numpy as np
import array
import sys
import statistics
import matplotlib.pyplot as plt



nb_points = 100
step = 1_000
STREAM_ARRAY_TYPE = float(0)
times = [0 for i in range(4)]
bandwidth_list = [0 for i in range(4)]
bandwidth_array = [0 for i in range(4)]
bandwidth_numpy = [0 for i in range(4)]


x_vector = np.logspace(0, 6, num=100, dtype = int)
print(x_vector)

data_list = [0 for i in range(nb_points)]
data_array = [0 for i in range(nb_points)]
data_numpy = [0 for i in range(nb_points)]


for i in range(nb_points):
    STREAM_ARRAY_SIZE = x_vector[i] 
    #Lists
    a = [1.0 for i in range(STREAM_ARRAY_SIZE)]
    b = [2.0 for i in range(STREAM_ARRAY_SIZE)]
    c = [0.0 for i in range(STREAM_ARRAY_SIZE)]
    scalar = 2.0
    
    # copy
    times[0] = timer()
    for j in range(STREAM_ARRAY_SIZE):
          c[j] = a[j]
    times[0] = timer() - times[0]
    
    # scale
    times[1] = timer()
    for j in range(STREAM_ARRAY_SIZE):
         b[j] = scalar*c[j]
    times[1] = timer() - times[1]
    #sum
    times[2] = timer()
    for j in range(STREAM_ARRAY_SIZE):
         c[j] = a[j]+b[j]
    times[2] = timer() - times[2]
    
    # triad
    times[3] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        a[j] = b[j]+scalar*c[j]
    times[3] = timer() - times[3]
    
    
    # Bandwith computation
    bandwidth_list[0] = 2 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[0]
    bandwidth_list[1] = 2 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[1]
    bandwidth_list[2] = 3 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[2]
    bandwidth_list[3] = 3 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[3]
    
    # print("Python List: ")
    # print("Bandwidth copy =", bandwidth_list[0], "bytes/sec")
    # print("Bandwidth scale =", bandwidth_list[1], "bytes/sec")
    # print("Bandwidth sum =", bandwidth_list[2], "bytes/sec")
    # print("Bandwidth triad =", bandwidth_list[3], "bytes/sec")
    
    data_list[i] = statistics.mean(bandwidth_list)
    
    #Arrays
    a = array.array('f', range(STREAM_ARRAY_SIZE))
    b = array.array('f', range(STREAM_ARRAY_SIZE))
    c = array.array('f', range(STREAM_ARRAY_SIZE))
    scalar = 2.0
    
    for j in range(STREAM_ARRAY_SIZE):
        a[j] = 1.0
        b[j] = 2.0
        c[j] = 0.0
    
    # copy
    times[0] = timer()
    for j in range(STREAM_ARRAY_SIZE):
          c[j] = a[j]
    times[0] = timer() - times[0]
    
    # scale
    times[1] = timer()
    for j in range(STREAM_ARRAY_SIZE):
         b[j] = scalar*c[j]
    times[1] = timer() - times[1]
    #sum
    times[2] = timer()
    for j in range(STREAM_ARRAY_SIZE):
         c[j] = a[j]+b[j]
    times[2] = timer() - times[2]
    
    # triad
    times[3] = timer()
    for j in range(STREAM_ARRAY_SIZE):
        a[j] = b[j]+scalar*c[j]
    times[3] = timer() - times[3]
    
    
    # Bandwith computation
    bandwidth_array[0] = 2 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[0]
    bandwidth_array[1] = 2 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[1]
    bandwidth_array[2] = 3 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[2]
    bandwidth_array[3] = 3 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[3]
    
    # print("Python array: ")
    # print("Bandwidth copy =", bandwidth_array[0], "bytes/sec")
    # print("Bandwidth scale =", bandwidth_array[1], "bytes/sec")
    # print("Bandwidth sum =", bandwidth_array[2], "bytes/sec")
    # print("Bandwidth triad =", bandwidth_array[3], "bytes/sec")
    
    data_array[i] = statistics.mean(bandwidth_array)
    
    #Numpy
    a = np.arange(STREAM_ARRAY_SIZE)
    b = np.arange(STREAM_ARRAY_SIZE)
    c = np.arange(STREAM_ARRAY_SIZE)
    scalar = 2.0
    
    for j in range(STREAM_ARRAY_SIZE):
        a[j] = 1.0
        b[j] = 2.0
        c[j] = 0.0
    
    # copy
    times[0] = timer()
    c = a
    times[0] = timer() - times[0]
    
    # scale
    times[1] = timer()
    b = scalar*c
    times[1] = timer() - times[1]
    #sum
    times[2] = timer()
    c = a + b 
    times[2] = timer() - times[2]
    
    # triad
    times[3] = timer()
    a = b + scalar*c
    times[3] = timer() - times[3]
    
    
    #Bandwith computation
    bandwidth_numpy[0] = 2 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[0]
    bandwidth_numpy[1] = 2 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[1]
    bandwidth_numpy[2] = 3 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[2]
    bandwidth_numpy[3] = 3 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[3]
    
    # print("Python NumPy: ")
    # print("Bandwidth copy =", bandwidth_numpy[0], "bytes/sec")
    # print("Bandwidth scale =", bandwidth_numpy[1], "bytes/sec")
    # print("Bandwidth sum =", bandwidth_numpy[2], "bytes/sec")
    # print("Bandwidth triad =", bandwidth_numpy[3], "bytes/sec")
    
    data_numpy[i] = statistics.mean(bandwidth_numpy)


plt.title("Bandwidth in function of type of data structure")
plt.xlabel("Size of the list")
plt.ylabel("Bandwidth in bytes/sec")
plt.plot(x_vector, data_list, label = "Python lists bandwidth")
plt.plot(x_vector, data_array, label = "Array module bandwidth")
plt.plot(x_vector, data_numpy, label = "Numpy module bandwidth")

plt.semilogx()
plt.semilogy()
plt.legend()
plt.show()
