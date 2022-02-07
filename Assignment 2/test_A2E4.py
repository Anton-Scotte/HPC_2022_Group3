# if __name__ == "__main__":
from A2E4 import list_DGEMM, array_DGEMM, np_array_DGEMM
import pytest
import numpy as np
import array


def test_list_DGEMM():
    A = [[5.2, 2.2, 1], [1.3, 2.2, 1.9], [1, 4.5, 3.0]]
    B = [[4.3, 2.3, 8.0], [1.3, 5.5, 7.3], [9.1, 5.3, 3.5]]
    C = [[6.2, 8.8, 6.4], [5.1, 2.0, 9.2], [0.7, 8.2, 4.8]]
    solution = [[40.52, 38.16, 67.56],
                [30.84, 27.16, 42.31],
                [38.15, 51.15, 56.15]]
    result = list_DGEMM(A, B, C)
    N = len(A)
    for i in range(N):
        for j in range(N):
            assert np.round(result[i][j], 2) == solution[i][j]


def test_array_DGEMM():
    A = [array.array('d', [5.2, 2.2, 1]),
         array.array('d', [1.3, 2.2, 1.9]),
         array.array('d', [1, 4.5, 3.0])]
    B = [array.array('d', [4.3, 2.3, 8.0]),
         array.array('d', [1.3, 5.5, 7.3]),
         array.array('d', [9.1, 5.3, 3.5])]
    C = [array.array('d', [6.2, 8.8, 6.4]),
         array.array('d', [5.1, 2.0, 9.2]),
         array.array('d', [0.7, 8.2, 4.8])]
    solution = [[40.52, 38.16, 67.56],
                [30.84, 27.16, 42.31],
                [38.15, 51.15, 56.15]]
    result = array_DGEMM(A, B, C)
    N = len(A)
    for i in range(N):
        for j in range(N):
            assert np.round(result[i][j], 2) == solution[i][j]


def test_np_array_DGEMM():
    # Numpy
    A = np.array([[5.2, 2.2, 1], [1.3, 2.2, 1.9], [1, 4.5, 3.0]])
    B = np.array([[4.3, 2.3, 8.0], [1.3, 5.5, 7.3], [9.1, 5.3, 3.5]])
    C = np.array([[6.2, 8.8, 6.4], [5.1, 2.0, 9.2], [0.7, 8.2, 4.8]])
    solution = [[40.52, 38.16, 67.56],
            [30.84, 27.16, 42.31],
            [38.15, 51.15, 56.15]]
    result = np_array_DGEMM(A, B, C)
    N = len(A)
    for i in range(N):
        for j in range(N):
            assert np.round(result[i, j],2) == solution[i][j]
