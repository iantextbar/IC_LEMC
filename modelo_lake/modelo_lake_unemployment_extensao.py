# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:15:45 2022

@author: Ian

Lake Unemployment Model Extensao

"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Funcoes

def lake_model_ext(emp, des, t_con, t_dem, t_e_merc, t_s_merc, tol = 1e-5):
    
    # Forca de trabalho total inicial
    N = emp + des
    
    # Taxa de crescimento da forca de trabalho
    g = t_e_merc - t_s_merc
    
    # Vetor da taxa de desemprego e emprego
    xt = np.array([[des/N],
                   [emp/N]])
    
    # Dinamica do modelo - como atualizar o numero de desempregados
    # e de empregados
    n_contratados = (1 - t_s_merc) * (1 - t_con) + t_e_merc
    demitidos = (1 - t_s_merc) * t_dem + t_e_merc
    contratados = (1 - t_s_merc) * t_con
    n_demitidos = (1 - t_s_merc) * (1 - t_dem)
    
    # Criando as matrizes de coeficientes para a atualizacao
    # do modelo dinamico
    A = np.array([[n_contratados, demitidos],
                  [contratados, n_demitidos]])
    
    A_hat = (1 / (1 + g)) * A
    
    # Inicializando o erro
    error = tol + 1
    
    # Inicializando listas e periodos
    t = 1
    U_vec = np.array([des])
    E_vec = np.array([emp])
    N_vec = np.array([N])
    u_vec = np.array([xt[0]])
    e_vec = np.array([xt[1]])
    
    # Roda a dinamica
    while error > tol:
        
        # Atualiza taxas
        xt_1 = A_hat @ xt
        error = np.max(np.abs(xt_1 - xt))
        xt = xt_1
        
        # Atualiza estoques
        N *= (1 + g)
        U = N * xt_1[0]
        E = N * xt_1[1]
        
        # Apensa novos valores
        u_vec = np.append(u_vec, xt_1[0])
        e_vec = np.append(e_vec, xt_1[1])
        U_vec = np.append(U_vec, U)
        E_vec = np.append(E_vec, E)
        N_vec = np.append(N_vec, N)
        
        t += 1
        
    return u_vec[-1], u_vec, e_vec, U_vec, E_vec, N_vec, t, A_hat

# Simulando um trabalhador individual - Cadeias de Markov

def stationary_distribution(t_con, t_dem):
    
    # Probabilidade de estar desempregado
    # e empregado no estado estacionario
    p_u = t_dem / (t_con + t_dem)
    p_e = 1 - p_u
    
    return p_u, p_e
    

def markov_chain_lake(p_u, p_e, t_con, t_dem, periodos = 1001):
    
    assert 0 <= p_u <= 1
    assert 0 <= p_e <= 1
    assert p_u + p_e == 1
    assert 0 <= t_con <= 1
    assert 0 <= t_dem <= 1
    
    # A distribuicao de probabilidade da primeira
    # variavel aleatoria da cadeia de markov
    dist_prob = np.array([p_u, p_e])
    
    # A matriz estocastica
    stoch_mat = np.array([[1 - t_con, t_con],
                          [t_dem, 1 - t_dem]])
    
    # Criando listas
    desempregado = np.array([])
    empregado    = np.array([])
    tempo_u_vec = np.array([])
    tempo_e_vec = np.array([])
    
    # Para cada periodo
    for i in range(1, periodos):
        
        # Gerar um valor aleatorio entre 0 e 1
        # segundo a uniforme
        rand = np.random.uniform(0, 1, 1)
        
        # Se o valor aleatorio for abaixo da probabilidade
        # de estar desempregado segundo a distribuicao de 
        # Xt apensa 1 a lista de desempregado e 0 a lista
        # de empregado
        if rand <= dist_prob[0]:
            
            desempregado = np.append(desempregado, 1)
            empregado = np.append(empregado, 0)
            
        # Caso contrario faz o inverso
        else:
            
            desempregado = np.append(desempregado, 0)
            empregado = np.append(empregado, 1)
        
        # Com base no periodo calcula a porcentagem de tempo
        # empregado ou desempregado e apensa na lista
        percent_des = sum(desempregado == 1) / i
        percent_emp = sum(empregado == 1) / i
        
        tempo_u_vec = np.append(tempo_u_vec, percent_des)
        tempo_e_vec = np.append(tempo_e_vec, percent_emp)
        
        dist_prob = dist_prob @ stoch_mat
        
    return tempo_u_vec, tempo_e_vec, desempregado, empregado

###############################################################################
# Aplicando as funcoes

# Variaveis

# Extensao 1
emp = 100
des = 30
t_con = 0.4
t_dem = 0.05
t_e_merc = 0.05
t_s_merc = 0.02

# Extensao 2

p_u = 0.3
p_e = 0.7
t_con2 = 0.4
t_dem2 = 0.16

vec = lake_model_ext(emp, des, t_con, t_dem, t_e_merc, t_s_merc)

individuo = markov_chain_lake(p_u, p_e, t_con2, t_dem2)

estado_estacionario = stationary_distribution(t_con2, t_dem2)

###############################################################################
# Estatica Comparativa

col_vec = ['darkred', 'darkorange', 'blueviolet', 'darkcyan', 'green']
ls_vec = ['solid', 'dashed', 'dotted', 'dashdot', '--']

# Mudancas na taxa de entrada no mercado e seu impacto na tragetoria
# da taxa de desemprego

t_ent_vec = np.linspace(0.000001, 0.2, 5)

for t in range(len(t_ent_vec)):
    
    res = lake_model_ext(emp, des, t_con, t_dem, t_ent_vec[t], t_s_merc)
    plt.plot(res[1], c = col_vec[t], ls = ls_vec[t],
             label = '$ a = {taxa:.2f} $'.format(taxa = t_ent_vec[t]))

plt.title('Taxa de Entrada no Mercado - Impacto na Taxa de Desmprego')
plt.ylabel('Taxa de Desemprego')
plt.xlabel('T')
plt.legend(loc = 'best')
plt.show()

# Mudancas na taxa de saida do mercado e seu impacto na tragetoria 
# da taxa de desemprego

t_sai_vec = np.linspace(0.000001, 0.2, 5)

for t in range(len(t_sai_vec)):
    
    res = lake_model_ext(emp, des, t_con, t_dem, t_e_merc, t_sai_vec[t])
    plt.plot(res[1], c = col_vec[t], ls = ls_vec[t],
             label = '$ b = {taxa:.2f} $'.format(taxa = t_sai_vec[t]))

plt.title('Taxa de Saída do Mercado - Impacto na Taxa de Desmprego')
plt.ylabel('Taxa de Emprego')
plt.xlabel('T')
plt.legend(loc = 'best')
plt.show()

# Mudancas na taxa de entrada no mercado e seu impacto na tragetoria
# da taxa de emprego

t_ent_vec = np.linspace(0.000001, 0.2, 5)

for t in range(len(t_ent_vec)):
    
    res = lake_model_ext(emp, des, t_con, t_dem, t_ent_vec[t], t_s_merc)
    plt.plot(res[2], c = col_vec[t], ls = ls_vec[t],
             label = '$ a = {taxa:.2f} $'.format(taxa = t_ent_vec[t]))

plt.title('Taxa de Entrada no Mercado - Impacto na Taxa de Emprego')
plt.ylabel('Taxa de Desemprego')
plt.xlabel('T')
plt.legend(loc = 'best')
plt.show()

# Mudancas na taxa de saida do mercado e seu impacto na tragetoria 
# da taxa de emprego

t_sai_vec = np.linspace(0.000001, 0.2, 5)

for t in range(len(t_sai_vec)):
    
    res = lake_model_ext(emp, des, t_con, t_dem, t_e_merc, t_sai_vec[t])
    plt.plot(res[2], c = col_vec[t], ls = ls_vec[t],
             label = '$ b = {taxa:.2f} $'.format(taxa = t_sai_vec[t]))

plt.title('Taxa de Saída do Mercado - Impacto na Taxa de Emprego')
plt.ylabel('Taxa de Emprego')
plt.xlabel('T')
plt.legend(loc = 'best')
plt.show()

# Mudancas na taxa de entrada no mercado e seu impacto na tragetoria
# do estoque de desempregados

t_ent_vec = np.linspace(0.000001, 0.2, 5)

for t in range(len(t_ent_vec)):
    
    res = lake_model_ext(emp, des, t_con, t_dem, t_ent_vec[t], t_s_merc)
    plt.plot(res[3], c = col_vec[t], ls = ls_vec[t],
             label = '$ a = {taxa:.2f} $'.format(taxa = t_ent_vec[t]))

plt.title('Taxa de Entrada no Mercado - Impacto no Número de Desempregados')
plt.ylabel('Número de Desempregados')
plt.xlabel('T')
plt.legend(loc = 'best')
plt.show()

# Mudancas na taxa de saida do mercado e seu impacto na tragetoria 
# do estoque de desempregados

t_sai_vec = np.linspace(0.000001, 0.2, 5)

for t in range(len(t_sai_vec)):
    
    res = lake_model_ext(emp, des, t_con, t_dem, t_e_merc, t_sai_vec[t])
    plt.plot(res[3], c = col_vec[t], ls = ls_vec[t],
             label = '$ b = {taxa:.2f} $'.format(taxa = t_sai_vec[t]))

plt.title('Taxa de Saída do Mercado - Impacto no Número de Desempregados')
plt.ylabel('Número de Desempregados')
plt.xlabel('T')
plt.legend(loc = 'best')
plt.show()

# Mudancas na taxa de entrada no mercado e seu impacto na tragetoria
# do estoque de Empregados

t_ent_vec = np.linspace(0.000001, 0.2, 5)

for t in range(len(t_ent_vec)):
    
    res = lake_model_ext(emp, des, t_con, t_dem, t_ent_vec[t], t_s_merc)
    plt.plot(res[4], c = col_vec[t], ls = ls_vec[t],
             label = '$ a = {taxa:.2f} $'.format(taxa = t_ent_vec[t]))

plt.title('Taxa de Entrada no Mercado- Impacto no Número de Empregados')
plt.ylabel('Número de Empregados')
plt.xlabel('T')
plt.legend(loc = 'best')
plt.show()

# Mudancas na taxa de saida do mercado e seu impacto na tragetoria 
# do estoque de desempregados

t_sai_vec = np.linspace(0.000001, 0.2, 5)

for t in range(len(t_sai_vec)):
    
    res = lake_model_ext(emp, des, t_con, t_dem, t_e_merc, t_sai_vec[t])
    plt.plot(res[4], c = col_vec[t], ls = ls_vec[t],
             label = '$ b = {taxa:.2f} $'.format(taxa = t_sai_vec[t]))

plt.title('Taxa de Saída do Mercado - Impacto no Número de Empregados')
plt.ylabel('Número de Empregados')
plt.xlabel('T')
plt.legend(loc = 'best')
plt.show()

###############################################################################
# Graficos Individuo

plt.plot(individuo[0], c = 'steelblue')
plt.hlines(estado_estacionario[0], 0, 1000, color = 'red', ls = '--')
plt.title('Porcentagem de Tempo Desempregado')
plt.xlabel('T')
plt.ylabel('Porcentagem')
plt.show()

plt.plot(individuo[1], c = 'steelblue')
plt.hlines(estado_estacionario[1], 0, 1000, color = 'red', ls = '--')
plt.title('Porcentagem de Tempo Empregado')
plt.xlabel('T')
plt.ylabel('Porcentagem')
plt.show()