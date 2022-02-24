# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:12:33 2022

@author: victo
"""

    
from distutils.core import setup 
from Cython.Build import cythonize 
import numpy



setup(ext_modules=cythonize("cythonfn.pyx", 
                            compiler_directives={"language_level":"3"}),
                            include_dirs=[numpy.get_include()])
