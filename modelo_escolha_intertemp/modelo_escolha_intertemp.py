# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:36:02 2021

@author: Ian

Modelo Escolha Intertemporal 

Objetivos: Criar funcoes que resolvam o problema da escolha
intertemporal de consumo e poupanca, em uma economia com
2 periodos, para diversas formas funcionais. Retornar o 
consumo nos dois periodos e a poupanca otima no periodo 1. 

"""
# Importando Pacotes 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Resolucao analitica para uma funcao utilidade
# u(c) = ln(c)

def escolha_intertemp_analitica(beta, dota1, dota2, juros):
    
    # Calculando a renda no periodo t
    # e os monomios vindos da substituicao
    # de ct+1 da equacao de Fisher na restricao
    # orcamentaria
    
    renda_em_t = dota1 + dota2 / (1 + juros)
    monomio1   = 1 / (1 + beta)
    monomio2   = beta * (1 + juros)
    
    # Calculando consumo em t e t+1 
    # e poupanca
    
    ct   = monomio1 * renda_em_t
    ct_1 = monomio2 * ct
    s    = dota1 - ct
    
    return ct, ct_1, s

# Resolucao utilizando representacoes matriciais do 
# sistema de equacoes e resolvendo por Leontief

def escolha_intertemp_alglin(beta, dota1, dota2, juros):
    
    renda_em_t = dota1 + dota2 / (1 + juros)
    # preco_relativo_consumo
    juros_intertemp = 1 / (1 + juros)
    
    # Vetor do resultado
    res = np.array([[0],
                    [renda_em_t]])
    
    
    # Matriz de coeficientes
    mat_coef = np.array([[beta, -juros_intertemp],
                         [1, juros_intertemp]])
    
    # Matriz inversa de coeficientes
    inv_mat_coef = np.linalg.inv(mat_coef)
    
    # Produto interno da matriz inversa com 
    # o vetor de resultados
    consumo = np.dot(inv_mat_coef, res)
    
    # Isolando os resultados
    ct = consumo[0][0]
    ct_1 = consumo[1][0]
    s = dota1 - ct
    
    return ct, ct_1, s

# Resolucao utilizando iteracao por um grid - "numerico forca bruta"
def escolha_intertemp_num(func, beta, dota1, dota2, juros,
                          grids, gride, gridtam = 1000, tol = 2):
    
    renda_em_t           = dota1 + dota2 / (1 + juros)
    juros_intertemp = 1 / (1 + juros)
    
    # Define o grid
    ct_vec   = np.linspace(grids, gride, gridtam)
    ct_1_vec = np.linspace(grids, gride, gridtam)
    
    # Define as variaveis de output
    ct   = 0
    ct_1 = 0
    
    # Para cada par ordenado no grid
    # grid manual - valor base e adicoes a esse valor ate
    # chegar ao valor que quero dentro do uma tolerancia - seria busca local
    for i in ct_vec:
        for j in ct_1_vec:
            
            # Calcula o valor da equacao de fisher e da
            # restricao orcamentaria
            
            # Buscar o maximo da funcao objetivo - usamos apenas a preferencia - nao precisamos da tolerancia
            
            euler = (beta * func(j)) / func(i) - juros_intertemp
            res_orc = i + (j * juros_intertemp) - renda_em_t
            
            # Caso as igualdades na equacao de fisher
            # e na restricao orcamentaria valham, preenche
            # as variaveis de output com as respostas e quebra 
            # o loop
            
            # Minimizar valor absoluto de fisher - como em ponto fixo

            # Tolerancias pequenas nao encontram solucao 
            
            # res_orc deve ser <= 0
            
            # ?cycle se res_orc > 0
            
            if round(euler, tol) == 0 and round(res_orc, tol) == 0:
                ct   = i
                ct_1 = j
                # continue
                break
    
    return ct, ct_1, dota1 - ct

a, b, c = escolha_intertemp_analitica(beta = 0.8,
                                      dota1 = 8,
                                      dota2 = 11,
                                      juros = 0.1)

d, e, f = escolha_intertemp_alglin(beta = 0.8,
                                   dota1 = 8,
                                   dota2 = 11,
                                   juros = 0.1)

def d_ln(n):
    return 1 / n

def d_crra(c, alfa = 0.5):
    return c ** (-alfa)

cta, ct_1a, sa = escolha_intertemp_num(func = d_ln,
                                beta = 0.8,
                                dota1 = 8,
                                dota2 = 11,
                                juros = 0.1,
                                grids = 7,
                                gride = 11)

ctb, ct_1b, sb = escolha_intertemp_num(func = d_crra,
                                beta = 0.8,
                                dota1 = 8,
                                dota2 = 11,
                                juros = 0.1,
                                grids = 7,
    
                                gride = 11)

# Graficos

# Grafico preferencias
x = np.linspace(1, 11, 1000)

ct_1 = lambda x: np.exp((4.042386470181375 - np.log(x)) / 0.8)

pref = []

for i in x:
    pref.append(ct_1(i))
    

plt.plot(x, pref, 'r+', c = 'red', label = '$\ln(c_t) + 0.8\ln(c_{t+1})$')
plt.legend(loc = 'upper left')
plt.title('Preferências do Consumidor', fontsize = 20)
plt.xlabel('$c_t$', fontsize = 15)
plt.ylabel('$c_{t+1}$', fontsize = 15)
plt.show()

# Grafico restricao orcamentaria
res_orc_ct_1 = 19.8 - 1.1*x

plt.plot(x, res_orc_ct_1, c = 'green', label = '$c_t + {c_{t+1}}/{1.1} - 18 = 0$')
plt.legend(loc = 'upper left')
plt.title('Restrição Orcamentária do Consumidor', fontsize = 20)
plt.xlabel('$c_t$', fontsize = 15)
plt.ylabel('$c_{t+1}$', fontsize = 15)
plt.show()

# Grafico restricao e preferencias
# Usar linhas ao inves de + 
# Limitar ylim a [0, 30]

plt.plot(x, res_orc_ct_1, c = 'green', label = '$c_t + {c_{t+1}}/{1.1} - 18 = 0$', lw = 6)
plt.plot(x, pref, 'r+' , c = 'red', label = '$\ln(c_t) + 0.8\ln(c_{t+1})$')
plt.plot(b, a,'go', c = 'black')
plt.legend(loc = 'upper left')
plt.text(b - 0.7, a + 3, '(8.8, 10)')
plt.title('Restrição Orcamentária e Preferências do Consumidor', fontsize = 20)
plt.xlabel('$c_t$', fontsize = 15)
plt.ylabel('$c_{t+1}$', fontsize = 15)
plt.show()

# Preferencias Contour

y = np.linspace(0.01, 11, 1000)

# Cria matrizes - x tera linhas com o mesmo valor
# y tera linhas em iguais a y
matriz_x = []
matriz_y = []
# Fixa o valor de x
for i in x:
    # Inicia as linhas das matrizes
    linha_x = []
    linha_y = []
    # Itera na lista de valores de y
    for j in y:
        # Apensa na linha da matriz x o valor fixado de x
        linha_x.append(i)
        # Apensa na linha da matriz y o valor de y da iteração
        linha_y.append(j)
    # Guarda a linha criada na lista da matriz
    matriz_x.append(linha_x)
    matriz_y.append(linha_y)

# Transforma listas em arrays
matriz_x = np.array(matriz_x)
matriz_y = np.array(matriz_y)

# Matriz de utilidades calculadas
pref = lambda x, y: np.log(x) + 0.8 * np.log(y)
util = pref(matriz_x, matriz_y)

plt.contour(matriz_x, matriz_y, util, levels = 30)
plt.title('Curvas de Indiferença', fontsize = 20)
plt.xlabel('$c_t$', fontsize = 15)
plt.ylabel('$c_{t+1}$', fontsize = 15)
plt.show()

# Alteracoes de B e o impacto em ct, ct+1 e s

ct_vec = []
ct1_vec = []
s_vec = []

beta = np.linspace(0, 1, 100)

for i in beta:
    ct, ct1, s = escolha_intertemp_alglin(i, 8, 11, 0.1)
    ct_vec.append(ct)
    ct1_vec.append(ct1)
    s_vec.append(s)
    
plt.plot(beta, s_vec, 'r^', label = 'Poupança')
plt.plot(beta, ct_vec, ls = 'dashed', label = '$c_t$')
plt.plot(beta, ct1_vec, c = 'green', label = '$c_{t+1}$')
plt.title('Impactos de Alterações em Beta', fontsize = 20)
plt.xlabel('Beta', fontsize = 15)
plt.ylabel('$c_t, c_{t+1}, s$', fontsize = 15)
plt.legend(loc = 'upper center', bbox_to_anchor = (1.25, 0.75))
plt.show()

# Alteracoes de juros e o impacto em ct, ct+1 e s
# Efeito renda / efeito substituicao - outras interpretacoes
# CRRA - mexendo na aversao ao risco alteramos resultados?

ct_vec = []
ct1_vec = []
s_vec = []

juros = np.linspace(0, 1, 100)

# Atualizar renda tambem - efeito renda

for i in juros:
    ct, ct1, s = escolha_intertemp_alglin(0.8, 8, 11, i)
    ct_vec.append(ct)
    ct1_vec.append(ct1)
    s_vec.append(s)
    
plt.plot(juros, s_vec, 'r^', label = 'Poupança')
plt.plot(juros, ct_vec, ls = 'dashed', label = '$c_t$')
plt.plot(juros, ct1_vec, c = 'green', label = '$c_{t+1}$')
plt.title('Impactos de Alterações nos Juros', fontsize = 20)
plt.xlabel('Taxa de Juros', fontsize = 15)
plt.ylabel('$c_t, c_{t+1}, s$', fontsize = 15)
plt.legend(loc = 'upper center', bbox_to_anchor = (1.25, 0.75))
plt.show()