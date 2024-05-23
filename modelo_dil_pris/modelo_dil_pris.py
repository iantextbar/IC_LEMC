# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 19:47:22 2021

@author: Ian

Stag Hunt 
"""
# Importando pacotes

import numpy as np

# Definindo matrizes que serao usadas para teste

payoff1 = np.array([[4, 1],
                   [3, 2]])

payoff2 = np.array([[4, 3],
                   [1, 2]])

pris1 = np.array([[-3, 0],
                  [-6, -1]])

pris2 = np.array([[-3, -6],
                  [0, -1]])

j1 = np.array([[2, 0],
              [0, 0]])

j2 = np.array([[1, 0],
              [0, 0]])

matrizes = [payoff1, payoff2]
mats = [pris1, pris2]
o_mats = [j1, j2]

# Definindo a funcao que encontra equilibrios puros

def eq_puros(mat_vec):
    
    """
    Encontra os equilibrios puros dado um vetor contendo
    as matrizes de payoffs
    """
    
    # Definindo listas de melhores jogadas
    # os payoffs e a lista de payoffs no
    # equilibrio
    best1 = []
    best2 = []
    payoff1_t = np.transpose(mat_vec[0])
    payoff2 = mat_vec[1]
    eqs_pay = []
    
    # Para cada linha do payoff 1 transposto e
    # do payoff 2 encontra a posicao do maior payoff
    
    # !Fazer funcionar para situacoes em que varias estrategias
    # !tem o mesmo payoff
    for line in range(len(mat_vec[0])):
        best1.append(np.argmax(payoff1_t[line]))
        best2.append(np.argmax(payoff2[line]))
        
    # Caso coincidirem locais com maior payoff
    # para ambos os jogadores isso sera um 
    # equilibrio puro
    for i in range(len(best1)):
        if best1[i] in best2:
            loc = best1[i]
            temp = [payoff1_t[loc][loc], payoff2[loc][loc]]
            eqs_pay.append(temp)
    
    for i in range(len(eqs_pay) - 1):
        for j in range(i + 1, len(eqs_pay)):
            if eqs_pay[i] == eqs_pay[j]:
                eqs_pay = eqs_pay.pop(j)
    
    print('Os equilíbrios possuem payoffs = ', eqs_pay)
    
    return eqs_pay

# Definindo a funcao que encontra equilibrios mistos
# Possibilitar que encontre equilibrio na situacao de 
# payoffs iguais

def eq_mis(mat_pay, t = 10000):
    
    """
    Encontra os equilibrios mistos dado uma matriz de payoffs
    em que as linhas representam os payoffs de uma jogada.
    """
    
    # Cria um vetor de probabilidades e diferencas
    p_vec = np.linspace(0, 1, t)
    dif = []
    
    # Para cada probabilidade
    for p in p_vec:
        
        # No equilibrio as esperancas de cada jogada
        # devem ser iguais
        probs = np.array([p, 1 - p])
        esp_1 = np.dot(mat_pay[0], probs)
        esp_2 = np.dot(mat_pay[1], probs)
        
        dif.append(abs(esp_1 - esp_2))
    
    p = p_vec[np.argmin(dif)]
    
    return p

# Cria a funcao que encontra os equilibrios puros e mistos
# para um jogo simples de dois jogadores
def equilibrios(mat_vec):
    
    eq_puros_pay = eq_puros(mat_vec)
    eq_m_1 = eq_mis(mat_vec[0])
    eq_m_2 = eq_mis(np.transpose(mat_vec[1]))
    
    eq_misto = {'jogador_1':[eq_m_1, 1 - eq_m_1],
              'jogador_2':[eq_m_2, 1 - eq_m_2]}
    
    return eq_puros_pay, eq_misto

eq1 = eq_puros(matrizes)
eq2 = eq_puros(mats)
eq3 = eq_puros(o_mats)

alfa = eq_mis(payoff1)

puros, mistos = equilibrios(matrizes)