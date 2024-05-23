# -*- coding: utf-8 -*-
"""
Modelo de Solow

Created on Mon Oct 25 14:23:56 2021

@author: Ian
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set

###############################################################################
# Modelo sem crescimento populacional nem tecnologia

def produto(K, L, exp, K_por_trab = True):
    
    """
    Uma funcao de producao Cobb-Douglas:
    ______________________________________
    K          = estoque de capital
    L          = forca de trabalho
    exp        = o expoente da funcao Cobb-Douglas
    K_por_trab = True -> a função produto se torna produto por trabalhador.
    Nesse caso recomenda-se usar L = 1.
    ______________________________________
    """
    
    if K_por_trab:
        return (K / L) ** exp
    else:
        return (K ** exp) * (L ** exp)
    
def estado_estacionario(K, exp, s, depr, tol = 4):
    
    """
    Encontra o equilibrio estacionario para um modelo com forca de
    trabalho constante e uma funcao de producao Cobb-Douglas. 
    ______________________________________
    K        = estoque de capital inicial
    exp      = o expoente da funcao Cobb-Douglas
    s        = taxa de poupanca (entre 0 e 1)
    depr     = taxa de depreciacao (entre 0 e 1)
    precisao = a quantas casas decimais se deseja arredondar
    ______________________________________
    """
    
    assert s >= 0 and s <= 1
    assert depr >= 0 and depr <= 1
    
    # Inicializa o estoque de capital
    # do periodo t+1 e a variacao no estoque
    k_t_mais = K
    delta = s * produto(K, 1, exp) - depr * K
    
    # Enquanto o investimento for diferente da 
    # depreciacao, atualiza o estoque de capital
    while round(delta, tol) != 0:
        k_t_mais = K + delta
        delta = s * produto(k_t_mais, 1, exp) - depr * k_t_mais
        K = k_t_mais
        
    return round(K, tol)

svec = np.linspace(0, 1, 1000)
teste = estado_estacionario(4, 0.5, 0.3, 0.1)

def regra_de_ouro(exp, depr, k_ini = 0.001, st = 1000, tol = 4):
    
    """
    Encontra a taxa de poupanca que maximiza o consumo
    ______________________________________
    exp   = expoente da cobb-douglas
    depr  = taxa de depreciacao
    k_ini = valor inicial para o estoque de capital
    st    = tamanho do vetor de taxas de poupanca
    tol   = valor de tolerancia para arredondamento (default = 4)
    ______________________________________
    """
    
    assert depr >= 0 and depr <= 1
    
    # Cria vetor de taxas de poupanca
    s_vec = np.linspace(0, 1, st)
    c_vec = []
    k_vec = []
    
    # Encontra o capital no estado estacionario
    # para cada valor de poupanca
    for s in s_vec:
        
        k = estado_estacionario(k_ini, exp, s, depr)
        c = produto(k, 1, exp) - depr * k
        k_vec.append(k)
        c_vec.append(c)
        
    # Encontra o consumo maximo e a taxa de 
    # poupanca e capital correspondentes
    c_max = max(c_vec)
    s_max = s_vec[np.argmax(c_vec)]
    k_max = k_vec[np.argmax(c_vec)]
    
    return c_max, s_max, k_max, c_vec, k_vec, s_vec
        
a = regra_de_ouro(0.5, 0.1)

###############################################################################
# Modelo com crescimento populacional

def estado_estacionario_cresc_pop(exp, s, depr, n, K, L, tol = 4):
    
    """
    Encontra o estado estacionario para uma economia com crescimento
    populacional
    ______________________________________
    exp  = o expoente da funcao cobb-douglas
    s    = a taxa de poupanca (entre 0 e 1)
    depr = a taxa de depreciacao (entre 0 e 1)
    n    = a taxa de crescimento populacional
    K    = o estoque de capital inicial
    L    = o numero de trabalhadores inicial
    ______________________________________
    """
    
    assert s >= 0 and s <= 1
    assert depr >= 0 and depr <= 1
    
    # Inicializa o estoque de capital
    # do periodo t+1 e a variacao no estoque
    k_t_mais = K
    delta = s * produto(K, L, exp) - (depr + n) * K
    
    # Enquanto o investimento for diferente da 
    # depreciacao, atualiza o estoque de capital
    while round(delta, tol) != 0:
        
        k_t_mais = K + delta
        delta = s * produto(k_t_mais, L, exp) - (depr + n) * k_t_mais
        K = k_t_mais
    
    return round(K, tol)

teste2 = estado_estacionario_cresc_pop(0.5, 0.3, 0.1, 0.01, 4, 100)

def regra_de_ouro_cresc_pop(exp, depr, n, k_ini, l_ini, st = 1000, tol = 4):
    
    """
    Encontra a taxa de poupanca que maximiza o consumo para
    uma economia com crescimento populacional
    ______________________________________
    exp = o expoente da funcao cobb-douglas
    depr = a taxa de depreciacao (entre 0 e 1)
    n = a taxa de crescimento populacional
    k_ini = valor inicial para o estoque de capital
    l_ini = valor inicial para o numero de trabalhadores
    st = tamanho do vetor de taxas de poupanca
    ______________________________________
    """
    
    assert depr >= 0 and depr <= 1
    
    # Criando o vetor de poupancas
    s_vec = np.linspace(0, 1, st)
    k_vec = []
    c_vec = []
    
    # Encontra o capital no estado estacionario
    # para cada valor de poupanca
    for s in s_vec:
        
        k = estado_estacionario_cresc_pop(exp, s, depr, n, k_ini, l_ini)
        c = produto(k, l_ini, exp) - (depr + n) * k
        k_vec.append(k)
        c_vec.append(c)
        
    c_max = max(c_vec)
    k_max = k_vec[np.argmax(c_vec)]
    s_max = s_vec[np.argmax(c_vec)]

    return c_max, s_max, k_max, c_vec, k_vec, s_vec

b = regra_de_ouro_cresc_pop(0.5, 0.1, 0.01, 4, 100)

################################################################################
# Modelo com tecnologia

def produto_com_tech(K, L, E, exp, k_por_trab_ef = True):
    
    if k_por_trab_ef:
        
        return (K / (L * E)) ** exp
    
    else:
        
        return K ** exp * L ** exp * E ** exp

def estado_estacionario_tech(exp, s, depr, n, g, K, L, E, tol = 4):
    
    assert s >= 0 and s <= 1
    assert depr >= 0 and depr <= 1
    
    k_t_mais = K
    delta = s * produto_com_tech(K, L, E, exp) - (depr + n + g) * K
    
    while round(delta, tol) != 0:
        
        k_t_mais = K + delta
        delta = s * produto_com_tech(k_t_mais, L, E, exp) - (depr + n + g) * K
        K = k_t_mais
        
    return round(K, tol)

def regra_de_ouro_tech(exp, depr, n, g, k_ini, l_ini, e_ini, st = 1000):
    
    assert depr >= 0 and depr <= 1
    
    s_vec = np.linspace(0, 1, st)
    k_vec = []
    c_vec = []
    
    for s in s_vec:
        
        k = estado_estacionario_tech(exp, s, depr, n, g, k_ini, l_ini, e_ini)
        c = produto_com_tech(k_ini, l_ini, e_ini, exp) - (depr + n + g) * k
        k_vec.append(k)
        c_vec.append(c)
    
    c_max = max(c_vec)
    k_max = k_vec[np.argmax(c_vec)]
    s_max = s_vec[np.argmax(c_vec)]
    
    return c_max, s_max, k_max, c_vec, k_vec, s_vec

################################################################################
# Gráficos

def graficos(K, L, exp, s, depr, x_start = 0, x_end = 10, x_size = 10000):
    
    # Producao Cobb-Douglas
    x = np.linspace(x_start, x_end, x_size)
    plt.plot(x, produto(x, L, exp), c = "magenta", label = "$y = k^{{%s}}$"%exp, linewidth = 3)
    plt.title('Produto Agregado - Cobb-Douglas', fontsize = 20)
    plt.xlabel('k - Estoque de Capital por Trabalhador', fontsize = 15)
    plt.ylabel('y - Produto por Trabalhador', fontsize = 15)
    plt.legend(loc = 'upper left')
    plt.show()

    # Producao Cobb-Douglas e investimento
    plt.plot(x, produto(x, L, exp), c = "magenta", label = "$y = k^{{%s}}$"%exp, linewidth = 3)
    plt.plot(x, produto(x, L, exp) * s, c = "green", label = "$i = {poup} \cdot y$".format(poup = s), linewidth = 3, ls = ':')
    plt.title('Produto e Investimento', fontsize = 20)
    plt.xlabel('k - Estoque de Capital por Trabalhador', fontsize = 15)
    plt.ylabel('y - Produto por Trabalhador', fontsize = 15)
    plt.legend(loc = 'upper left')
    plt.show()

    # Producao Cobb-Douglas, investimento e depreciacao
    plt.plot(x, produto(x, L, exp), c = "magenta", label = "$y = k^{{%s}}$"%exp, linewidth = 3)
    plt.plot(x, produto(x, L, exp) * s, c = "green", label = "$i = {poup} \cdot y$".format(poup = s), linewidth = 3, ls = ':')
    plt.plot(x, depr * x, c = "orange", label = "$\delta k $", linewidth = 3, ls = 'dashdot')
    plt.title('Produto, Investimento e Depreciação', fontsize = 20)
    plt.xlabel('k - Estoque de Capital por Trabalhador', fontsize = 15)
    plt.ylabel('y - Produto por Trabalhador', fontsize = 15)
    plt.legend(loc = 'upper left')
    plt.show()

    # Producao Cobb-Douglas, investimento, depreciacao e ponto estacionario

    est = estado_estacionario(K, exp, s, depr)

    plt.plot(x, produto(x, L, exp), c = "magenta", label = "$y = k^{{%s}}$"%exp, linewidth = 3)
    plt.plot(x, produto(x, L, exp) * s, c = "green", label = "$i = {poup} \cdot y$".format(poup = s), linewidth = 3, ls = ':')
    plt.plot(x, depr * x, c = "orange", label = "$\delta k $", linewidth = 3, ls = 'dashdot')
    plt.plot(est, depr * est, 'go', c = "black", label = 'Estado Estacionário')
    plt.title('Estado Estacionário', fontsize = 20)
    plt.text(est - 0.5, s *produto(est, L, exp) + 0.15, '{:.2f}'.format(est))
    plt.xlabel('k - Estoque de Capital por Trabalhador', fontsize = 15)
    plt.ylabel('y - Produto por Trabalhador', fontsize = 15)
    plt.legend(loc = 'upper left', fontsize = 10)
    plt.show()

    reg_de_ou = regra_de_ouro(exp, depr, K)
    
    plt.plot(reg_de_ou[-1], reg_de_ou[3], c = 'lightblue')
    plt.plot(reg_de_ou[1], reg_de_ou[0], 'go', c = 'black')
    plt.xlabel('Taxa de Poupança', fontsize = 15)
    plt.ylabel('Consumo', fontsize = 15)
    plt.text(reg_de_ou[1], reg_de_ou[0] + 0.1, '{reg:.2f}'.format(reg = reg_de_ou[0]))
    plt.title('Consumo no Estado Estacionário', fontsize = 20)
    plt.ylim([0, reg_de_ou[0] + 0.5])
    plt.show()
    
    plt.plot(reg_de_ou[-1], reg_de_ou[4], c = 'purple')
    plt.xlabel('Taxa de Poupança', fontsize = 15)
    plt.ylabel('Estoque de Capital', fontsize = 15)
    plt.title('Estoque de Capital no Estado Estacionário', fontsize = 20)
    plt.show()

graficos(6, 1, 1/3, 0.3, 0.1, x_end = 15)

def graficos_cresc_pop(K, L, exp, s, depr, n, x_start = 0, x_end = 10, x_size = 10000):
    
    # Producao Cobb-Douglas, investimento, depreciacao e ponto estacionario

    x = np.linspace(x_start, x_end, x_size)
    est = estado_estacionario_cresc_pop(exp, s, depr, n, K, L, tol = 6)

    plt.plot(x, produto(x, L, exp), c = "magenta", label = "$y = k^{{%s}}$"%exp, linewidth = 3)
    plt.plot(x, produto(x, L, exp) * s, c = "green", label = "$i = {poup} \cdot y$".format(poup = s), linewidth = 3, ls = ':')
    plt.plot(x, (depr + n) * x, c = "orange", label = "$(\delta + n) k $", linewidth = 3, ls = 'dashdot')
    plt.plot(est, (depr + n) * est, 'go', c = "black", label = 'Estado Estacionário')
    plt.title('Estado Estacionário', fontsize = 20)
    plt.text(est, s * produto(est, L, exp) * 1.1, '{:.2f}'.format(est))
    plt.xlabel('k - Estoque de Capital por Trabalhador', fontsize = 15)
    plt.ylabel('y - Produto por Trabalhador', fontsize = 15)
    plt.legend(loc = 'upper left', fontsize = 10)
    plt.show()

    reg_de_ou = regra_de_ouro_cresc_pop(exp, depr, n, K, L)
    
    plt.plot(reg_de_ou[-1], reg_de_ou[3], c = 'lightblue')
    plt.plot(reg_de_ou[1], reg_de_ou[0], 'go', c = 'black')
    plt.xlabel('Taxa de Poupança', fontsize = 15)
    plt.ylabel('Consumo', fontsize = 15)
    plt.text(reg_de_ou[1], reg_de_ou[0] + 0.1, '{reg:.2f}'.format(reg = reg_de_ou[0]))
    plt.title('Consumo no Estado Estacionário', fontsize = 20)
    plt.ylim([0, reg_de_ou[0] + 0.5])
    plt.show()
    
    plt.plot(reg_de_ou[-1], reg_de_ou[4], c = 'purple')
    plt.xlabel('Taxa de Poupança', fontsize = 15)
    plt.ylabel('Estoque de Capital', fontsize = 15)
    plt.title('Estoque de Capital no Estado Estacionário', fontsize = 20)
    plt.show()
    
graficos_cresc_pop(6, 5, 0.5, 0.4, 0.1, 0.01, x_end = 3)

def graficos_tech(K, L, E, exp, s, depr, n, g, x_start = 0, x_end = 10, x_size = 10000):
    
    # Producao Cobb-Douglas, investimento, depreciacao e ponto estacionario

    x = np.linspace(x_start, x_end, x_size)
    est = estado_estacionario_tech(exp, s, depr, n, g, K, L, E, tol = 6)

    plt.plot(x, produto(x, L, exp), c = "magenta", label = "$y = k^{{%s}}$"%exp, linewidth = 3)
    plt.plot(x, produto(x, L, exp) * s, c = "green", label = "$i = {poup} \cdot y$".format(poup = s), linewidth = 3, ls = ':')
    plt.plot(x, (depr + n + g) * x, c = "orange", label = "$(\delta + n + g) k $", linewidth = 3, ls = 'dashdot')
    plt.plot(est, (depr + n + g) * est, 'go', c = "black", label = 'Estado Estacionário')
    plt.title('Estado Estacionário', fontsize = 20)
    plt.text(est, s * produto(est, L, exp) * 1.1, '{:.2f}'.format(est))
    plt.xlabel('k - Estoque de Capital por Trabalhador Efetivo', fontsize = 15)
    plt.ylabel('y - Produto por Trabalhador Efetivo', fontsize = 15)
    plt.legend(loc = 'upper left', fontsize = 10)
    plt.show()

    reg_de_ou = regra_de_ouro_tech(exp, depr, n, g, K, L, E)
    
    plt.plot(reg_de_ou[-1], reg_de_ou[3], c = 'lightblue')
    plt.plot(reg_de_ou[1], reg_de_ou[0], 'go', c = 'black')
    plt.xlabel('Taxa de Poupança', fontsize = 15)
    plt.ylabel('Consumo', fontsize = 15)
    plt.text(reg_de_ou[1], reg_de_ou[0] + 0.1, '{reg:.2f}'.format(reg = reg_de_ou[0]))
    plt.title('Consumo no Estado Estacionário', fontsize = 20)
    plt.ylim([0, reg_de_ou[0] + 0.5])
    plt.show()
    
    plt.plot(reg_de_ou[-1], reg_de_ou[4], c = 'purple')
    plt.xlabel('Taxa de Poupança', fontsize = 15)
    plt.ylabel('Estoque de Capital', fontsize = 15)
    plt.title('Estoque de Capital no Estado Estacionário', fontsize = 20)
    plt.show()
    
graficos_tech(6, 5, 1.1, 0.5, 0.3, 0.1, 0.01, 0.02, x_end = 3)