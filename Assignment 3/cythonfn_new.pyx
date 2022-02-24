#cython: boundscheck=False
# cythonfn_new.pyx
def gauss_seidel(f):
	cdef unsigned int i, j
	newf = f.copy()
    
	for i in range(1,newf.shape[0]-1):
		for j in range(1,newf.shape[1]-1):
			newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] +
								   newf[i+1,j] + newf[i-1,j])

	return newf