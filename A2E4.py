import numpy as np
import random
import array


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
    A = np.matrix([[5, 2, 1], [1, 2, 1], [1, 4, 3]])
    B = np.matrix([[4, 2, 8], [1, 5, 7], [9, 5, 3]])
    C = np.matrix([[6, 8, 6], [5, 2, 9], [0, 8, 4]])
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

    # # Lists
    # rows, cols = (5, 5)
    # A = [[random.random() for e in range(rows)] for e in range(cols)]

    # # Numpy
    # print(np_array_DGEMM(A, B, C))
    print("Correct solution: \n", correct_solution)
