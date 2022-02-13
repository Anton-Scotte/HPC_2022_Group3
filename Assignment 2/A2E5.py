from operator import matmul
from turtle import shape
import numpy as np
import random
from timeit import default_timer as timer
import matplotlib.pyplot as plt


# Task 5.1 Develop a DFT in Python and a unit test with pytest for checking the correctness of the calculation. The data structures (lists, array, or NumPy) are of your choice.

# DFT( double* xr,  double* xi, double* Xr_o, double* Xi_o,  int N){
#     for (int k=0 ; k<N ; k++){
#         for (int n=0 ; n<N ; n++{
#              // Real part of X[k]
#              Xr_o[k] += xr[n] * cos(n * k * PI2 / N) + xi[n]*sin(n * k * PI2 / N);
#             // Imaginary part of X[k]
#             Xi_o[k] += -xr[n] * sin(n * k * PI2 / N) + xi[n] * cos(n * k * PI2 / N);
#        }
#  }


def DFT(x):
    xr = np.real(x)
    xi = np.imag(x)
    N = len(xr)
    Xr_o = np.zeros(N)
    Xi_o = np.zeros(N)
    pi = np.pi
    for k in range(N):
        for n in range(N):
            cos = np.cos(n * k * 2*pi / N)
            sin = np.sin(n * k * 2*pi / N)
            # Real part of X[k]
            Xr_o[k] += xr[n] * cos + xi[n]* sin
            # Imaginary part of X[k]
            Xi_o[k] += -xr[n] * sin + xi[n] * cos
    return np.array(Xr_o+1j*Xi_o).reshape((-1,1))

#@profile
def DFT_N_1024(x):
    xr = np.real(x)
    xi = np.imag(x)
    N = len(xr)
    Xr_o = np.zeros(N)
    Xi_o = np.zeros(N)
    pi = np.pi
    for k in range(N):
        for n in range(N):
            cos = np.cos(n * k * 2*pi / N)
            sin = np.sin(n * k * 2*pi / N)
            # Real part of X[k]
            Xr_o[k] += xr[n] * cos + xi[n]* sin
            # Imaginary part of X[k]
            Xi_o[k] += -xr[n] * sin + xi[n] * cos
    return np.array(Xr_o+1j*Xi_o).reshape((-1,1))

if __name__ == "__main__":
    #Task 5.3 - For fixed input size N=1024 do profiling
    N = 1024
    x = np.random.uniform(0,50,N)
    DFT_N_1024(x)
    # Initial tests
    # x = np.array([4,2,8,5,1],dtype='complex')
    # print(DFT(x))
    # print(np.fft.fft(x).reshape((-1,1)))

# Task 5.2. Measure the execution time, varying the input size from 8 to 1024 elements and plot it.
    N_set = range(8,1024,3)
    # N_set = range(8,200)
    times = np.zeros(len(N_set))
    for i,N in enumerate(N_set):
        x = np.random.uniform(0,50,N)
        start = timer()
        DFT(x)
        times[i] = timer()-start
        if i%50==0: print(f"Iteration {i}/{len(N_set)}.")

    plt.plot(N_set, times, label="Numpy with optimization")
    plt.xlabel("Size of vector")
    plt.ylabel("Time (s)")
    plt.semilogy()
    plt.legend()
    #plt.savefig('Assignment 2/runtimes_for_DFT_naive.png')
    plt.show()
