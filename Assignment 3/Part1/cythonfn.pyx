# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:11:44 2022

@author: victo
"""


from timeit import default_timer as timer
import numpy as np
import statistics
import array
cimport numpy as np 


def STREAM(nb_points):
    
    STREAM_ARRAY_TYPE = float(0)
    cdef list times = [0 for i in range(4)]
    cdef list bandwidth_list = [0 for i in range(4)]
    cdef list bandwidth_array = [0 for i in range(4)]
    cdef list bandwidth_numpy = [0 for i in range(4)]



    cdef int [:] x_vector = np.logspace(0, 6, num=nb_points, dtype = np.int32)
    cdef list data_list = [0 for i in range(nb_points)]
    cdef list data_array = [0 for i in range(nb_points)]
    cdef list data_numpy = [0 for i in range(nb_points)]

    cdef double scalar = 2.0
    cdef int STREAM_ARRAY_SIZE
    
    """
    cdef double[:] a_numpy
    cdef double[:] b_numpy
    cdef double[:] c_numpy
    """
    cdef int i=0
    
    for i in range(nb_points):
        STREAM_ARRAY_SIZE = x_vector[i] 
        #Lists
        a = [1.0 for i in range(STREAM_ARRAY_SIZE)]
        b = [2.0 for i in range(STREAM_ARRAY_SIZE)]
        c = [0.0 for i in range(STREAM_ARRAY_SIZE)]
        
        # copy
        times[0] = timer()
        for j in range(STREAM_ARRAY_SIZE):
              c[j] = a[j]
        times[0] = timer() - times[0]
        
        # scale
        times[1] = timer()
        for j in range(STREAM_ARRAY_SIZE):
             b[j] = scalar*c[j]
        times[1] = timer() - times[1]
        #sum
        times[2] = timer()
        for j in range(STREAM_ARRAY_SIZE):
             c[j] = a[j]+b[j]
        times[2] = timer() - times[2]
        
        # triad
        times[3] = timer()
        for j in range(STREAM_ARRAY_SIZE):
            a[j] = b[j]+scalar*c[j]
        times[3] = timer() - times[3]
        
        
        # Bandwith computation
        bandwidth_list[0] = 2 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[0]
        bandwidth_list[1] = 2 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[1]
        bandwidth_list[2] = 3 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[2]
        bandwidth_list[3] = 3 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[3]
        
        data_list[i] = statistics.mean(bandwidth_list)
        
        #Arrays
        a = array.array('f', range(STREAM_ARRAY_SIZE))
        b = array.array('f', range(STREAM_ARRAY_SIZE))
        c = array.array('f', range(STREAM_ARRAY_SIZE))
        
        for j in range(STREAM_ARRAY_SIZE):
            a[j] = 1.0
            b[j] = 2.0
            c[j] = 0.0
        
        # copy
        times[0] = timer()
        for j in range(STREAM_ARRAY_SIZE):
              c[j] = a[j]
        times[0] = timer() - times[0]
        
        # scale
        times[1] = timer()
        for j in range(STREAM_ARRAY_SIZE):
             b[j] = scalar*c[j]
        times[1] = timer() - times[1]
        #sum
        times[2] = timer()
        for j in range(STREAM_ARRAY_SIZE):
             c[j] = a[j]+b[j]
        times[2] = timer() - times[2]
        
        # triad
        times[3] = timer()
        for j in range(STREAM_ARRAY_SIZE):
            a[j] = b[j]+scalar*c[j]
        times[3] = timer() - times[3]
        
        
        # Bandwith computation
        bandwidth_array[0] = 2 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[0]
        bandwidth_array[1] = 2 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[1]
        bandwidth_array[2] = 3 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[2]
        bandwidth_array[3] = 3 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[3]
        
        data_array[i] = statistics.mean(bandwidth_array)
        
        #Numpy
        a_numpy = np.arange(STREAM_ARRAY_SIZE)
        b_numpy = np.arange(STREAM_ARRAY_SIZE)
        c_numpy = np.arange(STREAM_ARRAY_SIZE)
        
        for j in range(STREAM_ARRAY_SIZE):
            a_numpy[j] = 1.0
            b_numpy[j] = 2.0
            c_numpy[j] = 0.0
        
        # copy
        times[0] = timer()
        c_numpy = a_numpy
        times[0] = timer() - times[0]
        
        # scale
        times[1] = timer()
        b_numpy = scalar*c_numpy
        times[1] = timer() - times[1]
        #sum
        times[2] = timer()
        c_numpy = a_numpy + b_numpy
        times[2] = timer() - times[2]
        
        # triad
        times[3] = timer()
        a_numpy = b_numpy + scalar*c_numpy
        times[3] = timer() - times[3]
        
        
        #Bandwith computation
        bandwidth_numpy[0] = 2 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[0]
        bandwidth_numpy[1] = 2 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[1]
        bandwidth_numpy[2] = 3 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[2]
        bandwidth_numpy[3] = 3 * STREAM_ARRAY_TYPE.__sizeof__() * STREAM_ARRAY_SIZE / times[3]
        
        data_numpy[i] = statistics.mean(bandwidth_numpy)        
        
    return x_vector, data_list, data_array, data_numpy
