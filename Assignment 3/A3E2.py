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

    sizes = range(3,50,10)
    times = np.ones(len(sizes))
    for i,size_ in enumerate(sizes):
        # Initialize grid
        x = np.random.randint(1000, size=(size_,size_))
        # Set boundaries to zero
        x[:,np.r_[0,-1]]=0
        x[np.r_[0,-1],:]=0

        start = timer()
        for j in range(1000):    
            x = gauss_seidel(x)

        if (size_ % 10 == 0) or (size_ % 3 == 0):
            print(f'Grid size {size_} done')
        times[i] = timer()-start

    plt.plot(sizes,times)
    plt.xlabel("Grid size NxN")
    plt.ylabel('Times (s)')
    plt.show()
