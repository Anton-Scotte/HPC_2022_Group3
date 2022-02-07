#if __name__ == "__main__":
from A2E4 import list_DGEMM, array_DGEMM, np_array_DGEMM
import pytest
import numpy as np
import array

def test_list_DGEMM():
	A = [[5, 2, 1], [1, 2, 1], [1, 4, 3]]
	B = [[4, 2, 8], [1, 5, 7], [9, 5, 3]]
	C = [[6, 8, 6], [5, 2, 9], [0, 8, 4]]
	solution = [[37, 33, 63], [20, 19, 34], [35, 45, 49]]
	result = list_DGEMM(A, B, C)
	N = len(A)
	for i in range(N):
		for j in range(N):
			assert result[i][j] == solution[i][j]

def test_array_DGEMM():
	A = [[5, 2, 1], [1, 2, 1], [1, 4, 3]]
	B = [[4, 2, 8], [1, 5, 7], [9, 5, 3]]
	C = [[6, 8, 6], [5, 2, 9], [0, 8, 4]]
	solution = [[37, 33, 63], [20, 19, 34], [35, 45, 49]]
	result = array_DGEMM(A, B, C)
	N = len(A)
	for i in range(N):
		for j in range(N):
			assert result[i][j] == solution[i][j]

def test_np_array_DGEMM():
	# Numpy
	A = np.matrix([[5, 2, 1], [1, 2, 1], [1, 4, 3]])
	B = np.matrix([[4, 2, 8], [1, 5, 7], [9, 5, 3]])
	C = np.matrix([[6, 8, 6], [5, 2, 9], [0, 8, 4]])
	solution = C+np.matmul(A, B)
	result = np_array_DGEMM(A, B, C)
	N = len(A)
	for i in range(N):
		for j in range(N):
			assert result[i, j] == solution[i, j]
