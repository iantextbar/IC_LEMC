# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:02:16 2021

@author: Ian

Modelo Escolha Intertemporal com Risco

Objetivos: Criar funcoes que resolvam o problema da escolha
intertemporal de consumo e poupanca havendo risco, em uma
economia com 2 periodos. Queremos analisar a situação da compra de
seguros e da incerteza sobre a renda. Retornar o consumo nos
dois periodos e a poupanca otima no periodo 1.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Poupanca otima para renda incerta e uma funcao utilidade CRRA
def formula(beta, R, y1, y2, gama, p, poupanca = 0.00001):
    
    beta_r = beta * R
    menos_gama = -gama
    p1 = (y1 - poupanca)**(menos_gama)
    p2 = (1 - p) * (((y2)/(1 - p) + R * poupanca) ** menos_gama)
    p3 = p * ((R * poupanca) ** menos_gama)
    formula = p1 - beta_r * (p2 + p3)
    
    return formula
    
def poupanca_precaucional(beta, R, y1, y2, gama, p, precisao):
    
    poupanca = 0.00001
    
    while round(formula(beta, R, y1, y2, gama, p, poupanca), precisao) != 0.0:
        poupanca += 0.002
        
    return poupanca

# Situacao em que a pessoa compra uma certa quantidade de
# seguro para se precaver ao risco. 
# Qual a quantidade de seguro otima?

def seguro_otm(func, val, perda, gama, p, cs, ce, ct = 1000):
    
    # Criando os vetores do grid, da utilidade
    # e dos pares ordenados de cg e cb
    cgvec = cbvec = np.linspace(cs, ce, ct)
    pares_cons = []
    util = []
    
    # Para cada parordenado no grid
    for cg in cgvec:
        for cb in cbvec:
            
            # Calcula a restricao orcamentaria no ponto
            res_orc = (1 - gama) * cg + gama * cb - val + gama * perda
            
            # Se o ponto esta sobre ou abaixo da restricao
            # orcamentaria
            if res_orc <= 0:
                
                # Calcula e apenda a utilidade a lista
                u_c1 = (1 - p) * func(cg)
                u_c2 = p * func(cb)
                util.append(u_c1 + u_c2)
                pares_cons.append([cg, cb])

    util = np.array(util)
    
    u_max = util.max()
    cg_m, cb_m = pares_cons[np.argmax(util)]
    k_max = (cb_m - val + perda) / (1 - gama)
    
    return k_max, cg_m, cb_m, u_max

def seguro_analitico(d_g, d_b, p, g):
    
    # dotacao
    dota = d_g + (g / (1 - g)) * d_b
    
    # consumos
    cb = (dota * p * (1 - g)) / g
    cg = (g / (1 - g)) * ((1 - p) / p) * cb
    
    # utilidade
    u = (1 - p) * np.log(cg) + p * np.log(cb)
    
    # seguro otimo
    k = (cb - d_b) / (1 - g)
    
    return k, cg, cb, u


def d_ln(n):
    return np.log(n)

def d_crra(c, alfa = 0.5):
    return (c ** (-alfa)) / (1 - alfa)

k, cg, cb, u = seguro_otm(d_ln, 35000, 10000, 0.1, 0.01, 0.01, 39000, 3000)

ka, cga, cba, ua = seguro_analitico(35000, 25000, 0.01, 0.1)

# Graficos

# Restricao Orcamentaria
crv = np.linspace(0, 6000, 1000)
cbv = (34000 - 0.1*crv)/0.9

plt.plot(crv, cbv, c = 'blue', label = '$c_g = 37777.78 - cb/9$')
plt.legend(loc = 'upper right')
plt.title('Restrição Orcamentária', fontsize = 20)
plt.xlabel('$c_b$', fontsize = 15)
plt.ylabel('$c_g$', fontsize = 15)
plt.show()


# Preferencias

cb_pref = lambda cr: np.exp((10.5054421 - 0.01*np.log(cr))/0.99)

pref = []
for i in crv:
    pref.append(cb_pref(i))
    
plt.plot(crv, pref, c = 'red', label = '$u(c_g, c_b) = 0.99\ln{c_g} + 0.01\ln{c_b}$')
plt.legend(loc = 'best')
plt.title('Preferências', fontsize = 20)
plt.xlabel('$c_b$', fontsize = 15)
plt.ylabel('$c_g$', fontsize = 15)
plt.show()

# Preferencias e Res_Orc

plt.plot(crv, pref, c = 'red', label = '$u(c_g, c_b) = 0.99\ln{c_g} + 0.01\ln{c_b}$')
plt.plot(crv, cbv, c = 'blue', label = '$c_g = 37777.78 - cb/9$')
plt.plot(cb, cg, 'go', c = 'orange', label = 'Escolha')
plt.legend(loc = 'upper right')
plt.text(cb - 10, cg + 5, '(3400, 37400)')
plt.title('Restrição Orcamentária e Preferências', fontsize = 20)
plt.xlim([0, 5000])
plt.ylim([37000, 39000])
plt.xlabel('$c_b$', fontsize = 15)
plt.ylabel('$c_g$', fontsize = 15)
plt.show()

