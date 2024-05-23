# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 17:30:04 2021

@author: Ian

Caixa de Edgeworth
"""

import numpy as np
import matplotlib.pyplot as plt

# Utilizando a formula dada pelo Varian, com o bem 2 tendo
# seu preco como numerario
def preco_eq_analitico(exp1, exp2, w1a, w2a, w1b, w2b):
    
    p1 = (exp1 * w2a + exp2 * w2b) /\
        ((1 - exp1) * w1a + (1 - exp2) * w1b) 
        
    return p1

# Funcao demanda para Cobb-Douglas do tipo x1^a x2^(1-a)
# p2 eh o numerario
def demanda(p1, w1, w2, exp, p2 = 1, bem1 = True):
    
    """
    p1   = preco do bem 1
    w1   = dotacao do bem 1
    w2   = dotacao do bem 2
    exp  = expoente da Cobb-Douglas
    p2   = preco do bem 2 (default = 1)
    bem1 = caso True calcula-se a demanda do bem 1, \
        cc. demanda do bem 2
    
    """
    
    dota = p1 * w1 + p2 * w2
    
    if bem1:
        dem  = (exp * dota) / p1
        return dem
    
    dem = ((1 - exp) * dota) / p2
    
    return dem

# Funcao demanda excedente agregada
def dem_ex_agreg(dem_a, dem_b, wa, wb):
    
    """
    dem_a = demanda da pessoa A
    dem_b = demanda da pessoa B
    wa    = dotacao da pessoa A
    wb    = dotacao da pessoa B
    """
    
    return dem_a + dem_b - wa - wb

# Encontra o equilibrio tomado p2 como o numerario
def equilibrio(w1a, w2a, w1b, w2b, exp1, exp2, ps, pe, pt):
    
    """
    w1a  = dotacao do bem 1 da pessoa A
    w2a  = dotacao do bem 2 da pessoa A
    w1b  = dotacao do bem 1 da pessoa B
    w2b  = dotacao do bem 2 da pessoa B
    exp1 = expoente da Cobb-Douglas da pessoa A
    exp2 = expoente da Cobb-Douglas da pessoa B
    ps   = comeco do vetor de precos
    pe   = fim do vetor de precos
    pt   = tamanho do vetor de precos
    """
    
    # Declarando vetores
    p1_vec = np.linspace(ps, pe, pt)
    dem1_a_vec = []
    dem1_b_vec = []
    dem_ex_abs = []
    
    # Para cada preco possivel
    for p1 in p1_vec:
        
        # Calcula-se a demanda de cada agente
        dem_a = demanda(p1, w1a, w2a, exp1)
        dem_b = demanda(p1, w1b, w2b, exp2)
        dem1_a_vec.append(dem_a)
        dem1_b_vec.append(dem_b)
        
        # Calcula-se a demanda excedente agregada para
        # o bem 1
        dem_ex = dem_ex_agreg(dem_a, dem_b, w1a, w1b)
        dem_ex_abs.append(abs(dem_ex))

    # Preco de equilibrio do bem 1 no ponto de minimo
    # do valor absoluto da demanda excedente agregada
    p1_eq = p1_vec[np.argmin(dem_ex_abs)]
    
    # Encontra demandas
    dem_1a = dem1_a_vec[np.argmin(dem_ex_abs)]
    dem_1b = dem1_b_vec[np.argmin(dem_ex_abs)]
    dem_2a = demanda(p1_eq, w1a, w2a, exp1, bem1 = False)
    dem_2b = demanda(p1_eq, w1b, w2b, exp2, bem1 = False)
    
    return dem_1a, dem_1b, dem_2a, dem_2b, p1_eq
    
# Encontra pontos na curva de contrato
def curva_de_contrato(w1_total, w2_total, exp1, exp2, w1t, w2t):
    
    """
    w1_total = dotacao total do bem 1
    w2_total = dotacao total do bem 2
    exp1     = expoente da Cobb-Douglas da pessoa A
    exp2     = expoente da Cobb-Douglas da pessoa B
    w1t      = tamanho do vetor para w1
    w2t      = tamanho do vetor para w2
    """
    
    # Criando os vetores de dotacoes
    w1_vec = np.linspace(0, w1_total, w1t)
    w2_vec = np.linspace(0, w2_total, w2t)
    
    # Criando os vetores de precos e demandas
    p1_v     = []
    dem1_a_v = []
    dem1_b_v = []
    dem2_a_v = []
    dem2_b_v = []
    
    # Para cada ponto no grid
    for w2 in w2_vec:
        for w1 in w1_vec:
            
            w1b = w1_total - w1
            w2b = w2_total - w2
            
            # Encontra preco de equilibrio
            p1 = preco_eq_analitico(exp1, exp2, w1, w2, w1b, w2b)
            
            # Encontra demandas no equilibrio
            dem1_a = demanda(p1, w1, w2, exp1)
            dem2_a = demanda(p1, w1, w2, exp1, bem1 = False)
            dem1_b = demanda(p1, w1b, w2b, exp2)
            dem2_b = demanda(p1, w1b, w2b, exp2, bem1 = False)
            
            # Adiciona valores na lista
            p1_v.append(p1)
            dem1_a_v.append(dem1_a)
            dem2_a_v.append(dem2_a)
            dem1_b_v.append(dem1_b)
            dem2_b_v.append(dem2_b)
            
    return dem1_a_v, dem2_a_v, dem1_b_v, dem2_b_v, p1_v
    
# Construido analiticamente
def equilibrio_analitico(x, w1, w2, exp1, exp2):
    
    parcela1 = (exp2 * w2 - exp2 * w2 * exp1) * x
    parcela2 = (exp2 * x - exp1 * x) - (exp2 * w1 * exp1) + (exp1 * w1)

    return parcela1 / parcela2

p = preco_eq_analitico(0.6, 0.5, 10, 5, 2, 7)

vec = equilibrio(10, 5, 2, 7, 0.6, 0.5, 1, 1.5, 100)
vec2 = equilibrio(23, 33, 14, 32, 0.5, 0.5, 1.2, 2, 1000)

y1a = equilibrio_analitico(vec[0], 12, 12, 0.6, 0.5)
y2a = equilibrio_analitico(vec2[0], 37, 65, 0.5, 0.5)

curva_contrato = curva_de_contrato(12, 12, 0.6, 0.5, 100, 3)

cobb_douglas = lambda x1, x2, a: (x1**a) * (x2**(1 - a))

# Graficos

# Curva de Contrato

plt.plot(curva_contrato[0], curva_contrato[1])
plt.show()

# Grafico

x = np.linspace(0.0001, 12, 100)
y = np.linspace(0.0001, 12, 100)

X, Y = np.meshgrid(x, y)

Ua = cobb_douglas(X, Y, 0.6)
Ub = cobb_douglas(12 - X, 12 - Y, 0.5)

util_a = cobb_douglas(8.311284046692606, 7.191919191919192, 0.6)
util_b = cobb_douglas(3.696498054474708, 4.797979797979798, 0.5)

pref_inv = lambda x1, u, a: np.exp((np.log(u) - np.log(x1**a)) / (1 - a))


x2a = pref_inv(x, util_a, 0.6)
x2b = pref_inv(x, util_b, 0.5)

fig, axs = plt.subplots(tight_layout = True)
axs.plot(curva_contrato[0], curva_contrato[1])
axs.plot(vec[0], vec[2], 'go', c = 'black')
axs.plot(x, x2a, c = 'red')
axs.set_ylim(0, 12)
axs.set_xlim(0, 12)
axs2 = axs.twinx()
axs2.invert_xaxis()
axs2.set_ylim(0, 12)
axs3 = axs.twiny()
axs3.plot(x, x2b, c = 'yellow')
axs3.set_xlim(0, 12)
plt.gca().invert_yaxis()
plt.show()


fig, axs = plt.subplots(tight_layout = True)
axs.set_xlabel('$X_A$', fontsize = 25)
axs.set_ylabel('$Y_A$', fontsize = 25)
axs2 = axs.twinx()
or_ylim = axs.get_ylim()
axs2.set_ylim(or_ylim)
axs2.set_ylabel('$Y_B$', fontsize = 25)
axs2.invert_yaxis()
axs3 = axs.twiny()
or_xlim = axs.get_xlim()
axs3.set_xlim(or_xlim)
axs3.set_xlabel('$X_B$', fontsize = 25)
axs3.invert_xaxis()
axs.plot(curva_contrato[0], curva_contrato[1], c = 'orange')
axs.plot(vec[0], vec[2], 'go', c = 'black')
prefs1 = axs.contour(X, Y, Ua, levels = [util_a], linestyle = 'dotted')
prefs2 = axs.contour(X, Y, Ub, levels = [util_b], colors = 'green')
plt.show()