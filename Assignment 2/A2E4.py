from statistics import mean
import numpy as np
import random
import array
from timeit import default_timer as timer
import matplotlib.pyplot as plt


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
    N = len(A)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] = C[i, j] + A[i, k] * B[k, j]
    return C


if __name__ == "__main__":
    # Numpy
    A = np.array([[5, 2, 1], [1, 2, 1], [1, 4, 3]])
    B = np.array([[4, 2, 8], [1, 5, 7], [9, 5, 3]])
    C = np.array([[6, 8, 6], [5, 2, 9], [0, 8, 4]])
    correct_solution = C+np.matmul(A, B)
    print("Numpy: \n", np_array_DGEMM(A, B, C))
    # Lists
    A = [[5, 2, 1], [1, 2, 1], [1, 4, 3]]
    B = [[4, 2, 8], [1, 5, 7], [9, 5, 3]]
    C = [[6, 8, 6], [5, 2, 9], [0, 8, 4]]

    print("List: \n", list_DGEMM(A, B, C))
    # Arrays
    A = [array.array('i', [5, 2, 1]),
         array.array('i', [1, 2, 1]),
         array.array('i', [1, 4, 3])]
    B = [array.array('i', [4, 2, 8]),
         array.array('i', [1, 5, 7]),
         array.array('i', [9, 5, 3])]
    C = [array.array('i', [6, 8, 6]),
         array.array('i', [5, 2, 9]),
         array.array('i', [0, 8, 4])]
    print("Array: \n", array_DGEMM(A, B, C))
    print("Correct solution: \n", correct_solution)

    # Task 2: Increase matrix and report mean+std
    # # Lists
    size_m_max = 100
    size_range = range(3, size_m_max)
    times_list = [1]*(len(size_range))
    times_array = [1]*(len(size_range))
    times_np_array = [1]*(len(size_range))
    i = 0
    for size_m in size_range:

        # lists
        A = [[np.random.randint(0, 100) for e in range(size_m)]
             for e in range(size_m)]
        B = [[np.random.randint(0, 100) for e in range(size_m)]
             for e in range(size_m)]
        C = [[np.random.randint(0, 100) for e in range(size_m)]
             for e in range(size_m)]
        start = timer()
        list_DGEMM(A, B, C)
        times_list[i] = timer() - start

        # Numpy
        A = np.random.randint(0, 100, (size_m, size_m))
        B = np.random.randint(0, 100, (size_m, size_m))
        C = np.random.randint(0, 100, (size_m, size_m))

        start = timer()
        np_array_DGEMM(A, B, C)
        times_np_array[i] = timer() - start

        # Array:
        M = []
        for j in range(size_m):
            M.append(array.array('i', [random.randint(0, 100)
                                       for e in range(size_m)]))

        start = timer()
        array_DGEMM(A, B, C)
        times_array[i] = timer() - start
        i += 1

    print(f"Mean List: {mean(times_list)}, std: {np.std(times_list)}")
    print(f"Mean Array: {mean(times_array)}, std: {np.std(times_array)}")
    print(
        f"Mean Numpy Array: {mean(times_np_array)}, std: {np.std(times_np_array)}")

    plt.plot(size_range, times_list, label="List")
    plt.plot(size_range, times_array, label="Array")
    plt.plot(size_range, times_np_array, label="Numpy Array")
    plt.xlabel("Size of matrix (NxN)")
    plt.ylabel("Time (s)")
    plt.legend()
    # plt.show()
    plt.savefig('runtimes_for_DGEMMs.png')
