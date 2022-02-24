# import cythonfn_numpy

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def gauss_seidel(f):
    newf = f.copy()
    
    for i in range(1,newf.shape[0]-1):
        for j in range(1,newf.shape[1]-1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] +
                                   newf[i+1,j] + newf[i-1,j])
    
    return newf


if __name__ == '__main__':

    # # Test individuals
    # # Initialize grid
    # x_1 = np.random.uniform(0,100, size = (40,40))
    # x_2 = np.random.uniform(0,100, size = (40,40))
    # # Set boundaries to zero
    # x_1[:,np.r_[0,-1]]=0
    # x_1[np.r_[0,-1],:]=0
    # x_2[:,np.r_[0,-1]]=0
    # x_2[np.r_[0,-1],:]=0

    # # for i in range(1000):

    sizes = range(3,200,10)
    times = np.ones(len(sizes))
    for i,size_ in enumerate(sizes):
        # Initialize grid
        x = x = np.random.uniform(0,100, size = (size_,size_))
        # Set boundaries to zero
        x[:,np.r_[0,-1]]=0
        x[np.r_[0,-1],:]=0

        start = timer()
        for j in range(1000):    
            x = gauss_seidel(x)
        times[i] = timer()-start
        if (size_ % 10 == 0) or (size_ % 3 == 0):
            print(f'Grid size {size_} done')


    plt.plot(sizes,times)
    plt.xlabel("Grid size NxN")
    plt.ylabel('Times (s)')
    plt.savefig("Assignment 3/A3E2_task_1_plot.png")
    plt.show()
