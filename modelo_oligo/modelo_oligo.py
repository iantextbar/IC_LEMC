# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 22:26:07 2021

@author: Ian

Modelo de Cournot - Oligopolio
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def modelo_cournot_analitico(intr_dem, inc_dem, intr_c1, inc_c1, intr_c2, inc_c2):
    
    y1 = (intr_dem + inc_c2 - 2 * inc_c1) / (3 * inc_dem)
    
    y2 = (intr_dem - inc_dem * y1 - inc_c2) / 2 * inc_dem
    
    return y1, y2


def modelo_cournot_duo_alglin(intr_dem, inc_dem, intr_c1,
                          inc_c1, intr_c2, inc_c2):
    
    # Criando a matriz de coeficientes
    
    mat_coef = np.array([[1, 0.5],
                         [0.5, 1]])
    
    # Calculando a matriz inversa
    
    assert np.linalg.det(mat_coef) != 0
    
    inv_matcoef = np.linalg.inv(mat_coef)
    
    # Criando vetor dos resultados
    
    vec_resl = np.array([[intr_dem / (2 * inc_dem) - inc_c1 / (2 * inc_dem)],
                         [intr_dem / (2 * inc_dem) - inc_c2 / (2 * inc_dem)]])
    
    
    y_vec = np.dot(inv_matcoef, vec_resl)
    
    # Preco
    
    p = intr_dem - inc_dem * (sum(y_vec))
    
    
    return y_vec[0], y_vec[1], p

def modelo_cournot_njogadores_alglin(intr_dem, inc_dem, n_joga, c_marg_vec):
    
    assert len(c_marg_vec) == n_joga
    assert type(c_marg_vec).__module__ == np.__name__
    
    # Formando o vetor res
    
    vec_intr = [intr_dem / (2 * inc_dem) for i in range(n_joga)]
    res      = vec_intr - c_marg_vec * (1 / (2 * inc_dem))
    
    # Formando a matriz de coeficientes
    
    mat_coef = [[1 if i == j else 0.5 for i in range(n_joga)] for j in range(n_joga)]
    
    # Calculando a matriz inversa
    
    assert np.linalg.det(mat_coef) != 0
    
    inv_mat_coef = np.linalg.inv(mat_coef)
    
    # Encontrando as quantidades produzidas
    
    y_vec = np.dot(inv_mat_coef, res)
    
    # Encontrando o preco de equilibrio
    
    p = intr_dem - inc_dem * (sum(y_vec))
    
    return y_vec, p
    
    
m1 = modelo_cournot_analitico(200, 1, 0, 20, 0, 20)
m12 = modelo_cournot_duo_alglin(200, 1, 0, 20, 0, 20)
m2 = modelo_cournot_analitico(200, 1, 0, 10, 0, 20)
m22 = modelo_cournot_duo_alglin(200, 1, 0, 10, 0, 20)

e, f = modelo_cournot_njogadores_alglin(200, 1, 50, np.array([1 for i in range(50)]))

# Graficos
x_vec = np.linspace(0, 120, 1000)
y2_vec = 90 - (x_vec / 2)
y1_vec = 190 - 2 * x_vec

plt.plot(x_vec, y2_vec, c = 'blue', label = '$y2 = 90 - y1/2$')
plt.plot(x_vec, y1_vec, c = 'red', label = '$y1 = 95 - y2/2$')
plt.plot(m2[0], m2[1], 'go', c = 'orange')
plt.title('Modelo de Cournot', fontsize = 20)
plt.xlabel('y1', fontsize = 15)
plt.ylabel('y2', fontsize = 15)
plt.text(m2[0] - 5, m2[1] + 6, '(66.667, 56.667)')
plt.legend(loc = 'best')
plt.show()

# Efeito do aumento de custos para uma firma

custos_c2 = [5, 25, 45, 65]
cores     = ['blue', 'green', 'purple', 'lightblue']
cores2     = ['orange', 'pink', 'black', 'yellow']

plt.plot(x_vec, y1_vec, c = 'red', label = '$y1 = 95 - y2/2$')
plt.title('Alterando os Custos', fontsize = 20)
plt.xlabel('y1', fontsize = 15)
plt.ylabel('y2', fontsize = 15)
for c in range(len(custos_c2)):
    y2 = (200 - custos_c2[c] - x_vec) / 2
    plt.plot(x_vec, y2, c = cores[c])
    a, b = modelo_cournot_analitico(200, 1, 0, 10, 0, custos_c2[c])
    plt.plot(a, b, 'go', c = cores2[c], label = 'c = {c: .2f}, y1 = {a1: .2f}, y2 = {b2: .2f}'.format(c = custos_c2[c], a1 = a, b2 = b))

plt.legend(loc = 'upper center', bbox_to_anchor = (1.25, 0.75))
plt.show()

# Efeito do aumento do numero de firmas identicas para os precos e 
# quantidades

x = np.arange(1, 51, 1)
precos = []
yvec   = []

for i in x:
    cvec = np.array([10 for j in range(i)])
    y, p = modelo_cournot_njogadores_alglin(200, 1, i, cvec)
    precos.append(p)
    yvec.append(y[0])

plt.plot(x, precos, c = 'blue', label = 'Preços')
plt.plot(x, yvec, c = 'red', label = 'Quantidades')
plt.legend(loc = 'best')
plt.xlabel('Número de Firmas', fontsize = 15)
plt.ylabel('p/y', fontsize = 15)
plt.title('Efeito do Aumento do Número de Firmas em p e em y', fontsize = 20)
plt.show()