# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 13:39:50 2021

@author: Ian

MODELO DE LEONTIEF
"""

import numpy as np

alfa = .1
beta = .4
gama = .3

d1 = 5
d2 = 8
d3 = 0

vetor_const = [[d1],
               [d2],
               [d3]]

vetor_array = np.array(vetor_const)

matrix = [[(1 - alfa), 0, 0],
          [0, (1 - beta), 0],
          [0, 0, (1 - gama)]]

matrix_array = np.array(matrix)

inv_mat = np.linalg.inv(matrix_array)

res = np.dot(inv_mat, vetor_array) 