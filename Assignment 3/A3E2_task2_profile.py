import numpy as np

@profile
def gauss_seidel(f):
    newf = f.copy()
    
    for i in range(1,newf.shape[0]-1):
        for j in range(1,newf.shape[1]-1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] +newf[i+1,j] + newf[i-1,j])
    
    return newf


if __name__ == '__main__':

    # Initialize grid
    x = np.random.uniform(0,100, size = (100,100))
    # Set boundaries to zero
    x[:,np.r_[0,-1]]=0
    x[np.r_[0,-1],:]=0

    for i in range(1000):
        x = gauss_seidel(x)


