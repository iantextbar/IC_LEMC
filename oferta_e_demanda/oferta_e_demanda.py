# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:54:12 2021

@author: Ian
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Construiremos agora as funcoes do nosso programa
# Consiideraremos apenas oferta e demanda lineares

def demanda(x, intr_d = 6, incl_d = 2, inv = False):

    if inv:
        return (- x / incl_d) + (intr_d / incl_d)
    
    return intr_d - incl_d * x

def oferta(x, intr_o = 3, incl_o = 1, inv = False):
    
    if inv:
        return (x / incl_o) - (intr_o / incl_o)
    
    return intr_o + incl_o * x

def excesso_de_demanda_lin(intr_d, incl_d, intr_o, incl_o, p):
    
    """
    Retorna o excesso de demanda para um dado preco de oferta
    dados os interceptos e inclinacoes das funcoes oferta e demanda
    """
    
    dem = demanda(p, intr_d, incl_d)
    oft = oferta(p, intr_o, incl_o)
    
    return dem - oft

def equilibrio(intr_d, incl_d, intr_o, incl_o, grid_e, grid_s = 0, grid_tam = 1000):
    
    """
    intr_d   = intercepto da demanda
    incl_d   = inclinacao da demanda
    intr_o   = intercepto da oferta
    incl_o   = inclinacao da oferta
    grid_s   = comeco do grid (0 = default)
    grid_e   = fim do grid
    grid_tam = tamanho do grid (1000 = default)
    """
    
    # Criando o grid e a lista dos valores de excesso de demanda
    grid = np.linspace(grid_s, grid_e, grid_tam)
    val_preco_excesso = []
    
    # Preenchendo os valores de excesso de demanda
    for ponto in grid:
        val_preco_excesso.append(abs(excesso_de_demanda_lin(intr_d, incl_d, 
                                                    intr_o, incl_o, ponto)))
        
    # Transformando a lista em np.array
    val_preco_excesso = np.array(val_preco_excesso)
    
    # Encontrando o equilibrio
    preco_eq = grid[np.argmin(val_preco_excesso)]
    qtd_eq = demanda(intr_d, incl_d, preco_eq)
    
    return qtd_eq, preco_eq

###############################################################################

# Impostos
    
def equilibrio_com_imposto1(intr_d, incl_d, intr_o, incl_o, imp,
                           grid_e, grid_s = 0, grid_tam = 1000,
                           sobre_demanda = False):
    """
    intr_d   = intercepto da demanda
    incl_d   = inclinacao da demanda
    intr_o   = intercepto da oferta
    incl_o   = inclinacao da oferta
    grid_s   = comeco do grid (0 = default)
    grid_e   = fim do grid
    grid_tam = tamanho do grid (1000 = default)
    """
    
    # Criando o grid e a lista dos valores de excesso de demanda
    grid = np.linspace(grid_s, grid_e, grid_tam)
    val_preco_excesso = []
    
    # Garantindo que o imposto eh um valor entre 0 e 1
    assert imp >= 0 and imp <= 1
    
    if sobre_demanda:
        intr_d = (1 - imp) * intr_d
        for ponto in grid:
            val_preco_excesso.append(abs(excesso_de_demanda_lin(intr_d, incl_d, 
                                                    intr_o, incl_o, ponto)))
    else:
        intr_o = (1 - imp) * intr_o
        for ponto in grid:
            val_preco_excesso.append(abs(excesso_de_demanda_lin(intr_d, incl_d, 
                                                    intr_o, incl_o, ponto)))
        
    # Transformando a lista em np.array
    val_preco_excesso = np.array(val_preco_excesso)
    
    # Encontrando o equilibrio
    preco_eq = grid[np.argmin(val_preco_excesso)]
    qtd_eq = demanda(intr_d, incl_d, preco_eq)
    
    return qtd_eq, preco_eq



###############################################################################

# Criando alguns graficos para usar no texto

# Demanda
x = np.linspace(0, 10, 1000)
y = demanda(x, inv = True)

plt.plot(x, y, '-r', label = 'P = -Q/2 + 3', linewidth = 3.0)
plt.title('Demanda', fontsize = 25)
plt.xlim([0, 6])
plt.ylim([0, 3])
plt.xlabel('Quantidade demandada', fontsize = 15)
plt.ylabel('Preço', fontsize = 15)
plt.legend(loc = 'lower left')
plt.show()

# Oferta
x = np.linspace(0, 10, 1000)
y = oferta(x, inv = True)

plt.plot(x, y, '-r', label = 'P = Q - 3', linewidth = 3.0, c = 'blue' )
plt.title('Oferta', fontsize = 25)
plt.xlim([0, 6])
plt.ylim([-3, 3])
plt.xlabel('Quantidade ofertada', fontsize = 15)
plt.ylabel('Preço', fontsize = 15)
plt.legend(loc = 'upper left')
plt.show()

# Equilibrio
plt.plot(x, demanda(x, inv = True), '-r', label = 'P = -Q/2 + 3', linewidth = 3.0, c = 'purple')
plt.plot(x, oferta(x, inv = True), '-r', label = 'P = Q - 3', linewidth = 3.0, c = 'blue' )
plt.plot(4, 1, 'go', label = 'Ponto de Equilíbrio', c = 'black')
plt.title('Equilíbrio', fontsize = 25)
plt.xlim([0, 6])
plt.ylim([-3, 3])
plt.xlabel('Quantidade', fontsize = 15)
plt.ylabel('Preço', fontsize = 15)
plt.legend(loc = 'lower left')
plt.show()


# Imposto
a,b = equilibrio_com_imposto1(6, -2, 3, 1, 0.2, 5, sobre_demanda = True)

plt.plot(x, demanda(x, inv = True), '-r', label = 'P = -Q/2 + 3', linewidth = 3.0, c = 'purple')
plt.plot(x, demanda(x, intr_d = 4.8, inv = True), '-r', label = 'P = -Q/2 + 2.4', linewidth = 3.0, c = 'red')
plt.plot(x, oferta(x, inv = True), '-r', label = 'P = Q - 3', linewidth = 3.0, c = 'blue' )
plt.plot(4, 1, 'go', label = 'Ponto de Equilíbrio - Sem Imposto', c = 'black')
plt.plot(a, b, 'go', label = 'Ponto de Equilíbrio - Com Imposto', c = 'orange')
plt.title('Equilíbrio - Imposto de 20% sobre a Demanda', fontsize = 25)
plt.xlim([0, 6])
plt.ylim([-3, 3])
plt.xlabel('Quantidade', fontsize = 15)
plt.ylabel('Preço', fontsize = 15)
plt.legend(loc = 'lower right', fontsize = 10)
plt.show()

a,b = equilibrio_com_imposto1(6, -2, 3, 1, 0.2, 5, sobre_demanda = False)

plt.plot(x, demanda(x, inv = True), '-r', label = 'P = -Q/2 + 3', linewidth = 3.0, c = 'purple')
plt.plot(x, oferta(x, intr_o = 2.4, inv = True), '-r', label = 'P = Q - 1.2', linewidth = 3.0, c = 'red' )
plt.plot(x, oferta(x, inv = True), '-r', label = 'P = Q - 3', linewidth = 3.0, c = 'blue' )
plt.plot(4, 1, 'go', label = 'Ponto de Equilíbrio - Sem Imposto', c = 'black')
plt.plot(a, b, 'go', label = 'Ponto de Equilíbrio - Com Imposto', c = 'orange')
plt.title('Equilíbrio - Imposto de 20% sobre a Oferta', fontsize = 25)
plt.xlim([0, 6])
plt.ylim([-3, 3])
plt.xlabel('Quantidade', fontsize = 15)
plt.ylabel('Preço', fontsize = 15)
plt.legend(loc = 'lower right', fontsize = 10)
plt.show()
    
    
    
    
    

    

