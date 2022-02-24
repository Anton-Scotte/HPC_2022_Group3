# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:28:01 2022

@author: victo
"""


import matplotlib.pyplot as plt

import cythonfn 




def STREAM():
    x_vector, data_list, data_array, data_numpy = cythonfn.STREAM(100)
    
    plt.show()
    plt.plot(x_vector, data_list, label = "Python lists bandwidth")
    plt.plot(x_vector, data_array, label = "Array module bandwidth")
    plt.plot(x_vector, data_numpy, label = "Numpy module bandwidth")
    plt.title("Bandwidth in function of type of data structure")
    plt.xlabel("Size of the list")
    plt.ylabel("Bandwidth in bytes/sec")


    plt.semilogx()
    plt.semilogy()
    plt.legend()
    plt.show()
    
    print("x_vector", x_vector)
    print("data_list", data_list)
    print("data_array", data_array)
    print("data_numpy", data_numpy)
    
    
STREAM()