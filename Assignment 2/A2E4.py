from statistics import mean
import numpy as np
import random
import array
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import sys


def list_DGEMM(A, B, C):
    N = len(A)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] = C[i][j] + A[i][k] * B[k][j]
    return C


def array_DGEMM(A, B, C):
    N = len(A)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i][j] = C[i][j] + A[i][k] * B[k][j]
    return C


def np_array_DGEMM(A, B, C):
    # N = len(A)
    # for i in range(N):
    #     for j in range(N):
    #         for k in range(N):
    #             C[i, j] = C[i, j] + A[i, k] * B[k, j]
    return C+np.matmul(A,B)


if __name__ == "__main__":
    # Numpy
    A = np.array([[5.2, 2.2, 1], [1.3, 2.2, 1.9], [1, 4.5, 3.0]])
    B = np.array([[4.3, 2.3, 8.0], [1.3, 5.5, 7.3], [9.1, 5.3, 3.5]])
    C = np.array([[6.2, 8.8, 6.4], [5.1, 2.0, 9.2], [0.7, 8.2, 4.8]])
    
    print(sys.getsizeof(A))
    correct_solution = C+np.matmul(A, B)
    print("Numpy: \n", np_array_DGEMM(A, B, C))
    # Lists
    A = [[5.2, 2.2, 1], [1.3, 2.2, 1.9], [1, 4.5, 3.0]]
    B = [[4.3, 2.3, 8.0], [1.3, 5.5, 7.3], [9.1, 5.3, 3.5]]
    C = [[6.2, 8.8, 6.4], [5.1, 2.0, 9.2], [0.7, 8.2, 4.8]]
    print(sys.getsizeof(A))
    print("List: \n", list_DGEMM(A, B, C))
    # Arrays
    A = [array.array('d', [5.2, 2.2, 1]),
         array.array('d', [1.3, 2.2, 1.9]),
         array.array('d', [1, 4.5, 3.0])]
    B = [array.array('d', [4.3, 2.3, 8.0]),
         array.array('d', [1.3, 5.5, 7.3]),
         array.array('d', [9.1, 5.3, 3.5])]
    C = [array.array('d', [6.2, 8.8, 6.4]),
         array.array('d', [5.1, 2.0, 9.2]),
         array.array('d', [0.7, 8.2, 4.8])]
    print("Array: \n", array_DGEMM(A, B, C))
    print(sys.getsizeof(A))
    print("Correct solution: \n", correct_solution)

    # Task 2: Increase matrix and report mean+std
    # # Lists
    size_m_max = 200
    size_range = range(3, size_m_max,3)
    times_list = [1]*(len(size_range))
    times_array = [1]*(len(size_range))
    times_np_array = [1]*(len(size_range))
    i = 0
    for size_m in size_range:

        # lists
        A = [[np.random.rand()*100 for e in range(size_m)]
             for e in range(size_m)]
        B = [[np.random.rand()*100 for e in range(size_m)]
             for e in range(size_m)]
        C = [[np.random.rand()*100 for e in range(size_m)]
             for e in range(size_m)]
        start = timer()
        list_DGEMM(A, B, C)
        times_list[i] = timer() - start
        i += 1
        if i % 50 == 0:
            print(f"List size {i} done.")
    i=0
    for size_m in size_range:
        # Numpy
        A = np.random.rand(size_m, size_m)*100
        B = np.random.rand(size_m, size_m)*100
        C = np.random.rand(size_m, size_m)*100

        start = timer()
        np_array_DGEMM(A, B, C)
        times_np_array[i] = timer() - start
        i += 1
        if i % 50 == 0:
            print(f"Numpy size {i} done.")
    i=0
    for size_m in size_range:    
        # Array:
        A = []
        B = []
        C = []
        for j in range(size_m):
            A.append(array.array('d', [np.random.randint(0, 100)
                                       for e in range(size_m)]))
            B.append(array.array('d', [np.random.randint(0, 100)
                        for e in range(size_m)]))
            C.append(array.array('d', [np.random.randint(0, 100)
                        for e in range(size_m)]))
        start = timer()
        array_DGEMM(A, B, C)
        times_array[i] = timer() - start
        i += 1
        if i % 50 == 0:
            print(f"Array size {i} done.")

    print(f"Mean List: {mean(times_list)}, std: {np.std(times_list)}")
    print(f"Mean Array: {mean(times_array)}, std: {np.std(times_array)}")
    print(
        f"Mean Numpy Array: {mean(times_np_array)}, std: {np.std(times_np_array)}")

    plt.plot(size_range, times_list, label="List")
    plt.plot(size_range, times_array, label="Array")
    plt.plot(size_range, times_np_array, label="Numpy Array")
    plt.xlabel("Size of matrix (NxN)")
    plt.ylabel("Time (s)")
    plt.semilogy()
    plt.legend()
    plt.savefig('Assignment 2/runtimes_for_DGEMMs_vectorization.png')
    plt.show()

    # Task 4.3 L1 Cache

    # Cache sizes for Johannes:
    # L1 = 128KB
    # L2 = 512KB
    # L3 = 3MB

    # number of grid elements we can store in total (128 kilobytes) / (64 bits) = 16 000
    # We have 3 Matrices -> 16 000 / 3 = 5333 elements per matrix fit into the cache.
    # sqrt(5333) = 73 is the number of rows/cols per matrix that could fit into the cache.
    # Though, there are probably other processes running thus we can probably fit only about three 50x50 matrices in L1.
    # Afterwards Numpy implementation starts getting slower than others.
    # At about N=44, there is a spike on the numpy implementation, could be due to L1 cache not being big enough.


    # Task 4.4
    # N^3 iterations per function call
    # assignment, addition and multiplication = 3 FLOPS per iteration
    # -> 3*N^3 FLOPS per function call

    # For Numpy implementation, there are 2N^2+N^2 (matmul = 2n^2-n^2, addition n^2, assignment n^2) FLOPS per function call

    # plt.plot([3*x**3 for x in size_range], times_list, label="List")
    # plt.plot([3*x**3 for x in size_range], times_array, label="Array")
    # plt.plot([3*x**3 for x in size_range], times_np_array, label="Numpy Array")
    # plt.ylabel("Time (s)")
    # plt.xlabel("FLOPS")
    # plt.legend()
    # # plt.savefig('Assignment 2/runtimes_for_DGEMMs.png')
    # plt.show()
    print("List FLOPS/s: ",np.mean(np.array([3*x**3 for x in size_range]) / np.array(times_list)))
    print("Array FLOPS/s: ",np.mean(np.array([3*x**3 for x in size_range]) / np.array(times_array)))
    print("Numpy Array FLOPS/s: ",np.mean(np.array([2*x^2+x^2 for x in size_range]) / np.array(times_np_array)))

    # Johannes' clock frequency is 2.5 GHz (boost 2.7GHz)

    # Mean List: 0.5003616742424241, std: 0.5592229448440301
    # Mean Array: 0.7102736909090908, std: 0.8121135599432922
    # Mean Numpy Array: 0.00016189242424244102, std: 0.00015840794699113962
    # List FLOPS/s:  11418098.445513315
    # Array FLOPS/s:  8670588.630288439
    # Numpy Array FLOPS/s:  2427991.4751817994

    # Thus, 
    # List FLOPS/s / Clock Frq:  11418098.445513315/2 500 000 000 = 0.005
    # Array FLOPS/s / Clock Frq:  8670588.630288439/2 500 000 000 = 0.003
    # Numpy Array FLOPS/s / Clock Frq:  2427991.4751817994/2 500 000 000 = 0.001

    # The measured FLOPS/s is only about 4% of the theoretical limit of the computer.
    #For numpy implementation, the FLOPS/s is much lower, but this is due to it making significantly less operations (vectorization)