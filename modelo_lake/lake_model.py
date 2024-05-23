# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 19:18:15 2021

@author: Ian

Lake Unemployment Model

"""
# Quero ter no final a 1) taxa natural de desemprego, 
# 2) a qtd de trabalhadores empregados e desempregados no estado
# estacionario, 3) a tragetoria dos empregados e desempregados
# ate o estado estacionario e o 4) numero de periodos ate o estado
# estacionario. 

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d

def taxa_natural_desemprego(t_con, t_dem):
    """
    Calcula a taxa natural de desemprego
    ___________________________________
    t_con: Taxa de contratacao
    t_dem: Taxa de demissao
    """
    return (t_dem / (t_con + t_dem))

# Atualiza o numero de trabalhadores empregados para 
# o periodo t + 1
def atualiza_empregados(emp_em_t, demitidos, contratados):
    return emp_em_t + contratados - demitidos


def atualiza_desempregados(des_em_t, demitidos, contratados):
    return des_em_t + demitidos - contratados

def lake_model(emp, des, t_con, t_dem, tol = 4):
    
    """
    Encontra a tragetoria do mercado de trabalho ate o estado estacionario
    ___________________________________
    emp   = Numero de empregados
    des   = Numero de desempregados
    t_con = Taxa de contratacao
    t_dem = Taxa de demissão
    tol   = A precisao da estimativa em casas decimais (default = 4)
    """
    
    # Numero de trabalhadores na economia
    L = emp + des
    
    # Encontra taxa natural de desemprego
    t_des_nat = taxa_natural_desemprego(t_con, t_dem)
    
    # Tragetoria empregados, desempregados e taxa de desemprego
    vec_emp = np.array([emp])
    vec_des = np.array([des])
    vec_t   = np.array([des/L])
    
    # Numbero de pessoas demitidas e contratadas
    demitidos = t_dem * vec_emp[-1]
    contratados = t_con * vec_des[-1]
    
    # Contador
    counter = 1
    
    # Encontra o estado de equilibrio
    while np.round(np.abs(contratados - demitidos), tol) > 10 ** (-tol):
        
        # Calcula numero de desempregados e empregados
        # no proximo periodo
        emp_t_mais_1 = atualiza_empregados(vec_emp[-1],
                                           demitidos,
                                           contratados)
        
        des_t_mais_1 = atualiza_desempregados(vec_des[-1],
                                              demitidos,
                                              contratados)
        
        vec_emp = np.append(vec_emp, emp_t_mais_1)
        vec_des = np.append(vec_des, des_t_mais_1)
        vec_t   = np.append(vec_t, des_t_mais_1 / L)
        
        # Atualiza o numero de contratados e demitidos
        demitidos = t_dem * vec_emp[-1]
        contratados = t_con * vec_des[-1]
        
        # Atualiza o counter
        counter += 1
    
    return t_des_nat, vec_emp, vec_des, vec_t, counter
  
un, trag_emp, trag_des, taxa_des, counter = lake_model(100, 30, 0.4, 0.01)
a = lake_model(100, 30, 0.4, 0.05)
a2 = lake_model(100, 30, 0.4, 0.1)
a3 = lake_model(100, 30, 0.4, 0.15)
a4 = lake_model(100, 30, 0.4, 0.2)

b = lake_model(100, 30, 0.45, 0.01)
b2 = lake_model(100, 30, 0.5, 0.01)
b3 = lake_model(100, 30, 0.55, 0.01)
b4 = lake_model(100, 30, 0.6, 0.01)


# Graficos Primeiros Valores

# Taxa de Desemprego
plt.plot(taxa_des, c = 'orange', label = 'c = 0.4 e d = 0.01')
plt.title('Taxas de Desemprego', fontsize = 20)
plt.xlabel('T', fontsize = 15)
plt.ylabel('$U/L$', fontsize = 15)
plt.legend(loc = 'upper right')
plt.show()

# Trajetoria Emprego 
plt.plot(trag_emp, c = 'blue', label = 'c = 0.4 e d = 0.01')
plt.title('Empregados', fontsize = 20)
plt.xlabel('T', fontsize = 15)
plt.ylabel('$E$', fontsize = 15)
plt.legend(loc = 'upper right')
plt.show()

# Trajetoria Desemprego
plt.plot(trag_des, c = 'red', label = 'c = 0.4 e d = 0.01')
plt.title('Desempregados', fontsize = 20)
plt.xlabel('T', fontsize = 15)
plt.ylabel('$E$', fontsize = 15)
plt.legend(loc = 'upper right')
plt.show()

# Graficos Estatica Comparativa
# Taxa de Desemprego
plt.plot(taxa_des, c = 'orange', label = 'd = 0.01')
plt.plot(a[3], c = 'red', label = 'd = 0.05', ls = ':')
plt.plot(a2[3], c = 'green', label = 'd = 0.1', ls = 'dashdot')
plt.plot(a3[3], c = 'purple', label = 'd = 0.15', ls = 'dashed')
plt.plot(a4[3], c = 'lightblue', label = 'd = 0.2', ls = '-.')
plt.title('Taxas de Desemprego no Tempo com c = 0.4', fontsize = 17)
plt.xlabel('T', fontsize = 15)
plt.ylabel('$U/L$', fontsize = 15)
plt.legend(loc = 'upper right')
plt.show()

plt.plot(taxa_des, c = 'pink', label = 'c = 0.4')
plt.plot(b[3], c = 'seagreen', label = 'c = 0.45', ls = ':')
plt.plot(b2[3], c = 'navy', label = 'c = 0.5', ls = 'dashdot')
plt.plot(b3[3], c = 'lawngreen', label = 'c = 0.55', ls = 'dashed')
plt.plot(b4[3], c = 'maroon', label = 'c = 0.6', ls = '-.')
plt.title('Taxas de Desemprego no Tempo com d = 0.01', fontsize = 17)
plt.xlabel('T', fontsize = 15)
plt.ylabel('$U/L$', fontsize = 15)
plt.legend(loc = 'upper right')
plt.show()

# Taxa Natural de Desemprego
d = np.linspace(0.0001, 1, 100)
c = np.linspace(0.0001, 1, 100)
C, D = np.meshgrid(c, d)
Z = taxa_natural_desemprego(C, D)
axs = plt.axes(projection = '3d')
axs.plot_surface(C, D, Z, cmap = 'winter', rstride = 1, cstride = 1)
axs.set_title('Taxa Natural de Desemprego', fontsize = 16)
axs.set_xlabel('Taxa de Contratação', fontsize = 10)
axs.set_ylabel('Taxa de Demissão', fontsize = 10)
axs.view_init(20, 340)

fig, ax = plt.subplots()
ax.contour(C, D, Z)
ax.set_title('Taxa Natural de Desemprego', fontsize = 20)
ax.set_xlabel('Taxa de Contratação', fontsize = 15)
ax.set_ylabel('Taxa de Demissão', fontsize = 15)
plt.show()

# Fazer um scatterplot de tempo de convergencia