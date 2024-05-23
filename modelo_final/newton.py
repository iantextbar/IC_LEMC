# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 20:01:29 2022

@author: Ian
"""
import numpy as np

def exp(x, k, l, m):
    
    return (x[0] ** 3) + (2 * x[1] ** 2) + x[0] + k * l * m

def aprox_dif(f, x, var, tol = 4, **params):
    
    # Iterador
    i = 1
    
    h = 0.1 ** i
    
    # Guardando o vetor x inicial
    x_vec_i = x.copy()
    
    # Criando o vetor x novo
    x_vec_n = x.copy()
    x_vec_n[var] = x_vec_n[var] + h
    
    # Aproximacao da derivada em x para o
    # valor inicial de h
    dif = (f(x_vec_n, **params) - f(x_vec_i, **params)) / h
    
    # Variavel que armazena os valores anteriores
    # de dif
    temp = 0
    
    while round(abs(dif - temp), tol) != 0:
        
        print(dif)
        print(h)
        
        # Reduzo h
        i += 1
        h = 0.1 ** i
        
        # Atualizo dif
        temp = dif
        
        # Atualizo o valor da variavel
        x_vec_n = x.copy()
        x_vec_n[var] = x_vec_n[var] + h
        
        # Aproximacao da derivada em x para o
        # valor inicial de h
        dif = (f(x_vec_n, **params) - f(x_vec_i, **params)) / h
        
    
    return dif


a = aprox_dif(exp, [2, 1], 1, 6, k = 1, l = 2, m = 3)