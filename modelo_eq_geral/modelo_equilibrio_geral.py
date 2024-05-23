# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 17:14:09 2022

@author: Ian

Equilibrio Geral
"""
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Consumidor ----

# Preferencias
cobb_douglas = lambda x, laz, exp: x ** exp * laz ** (1 - exp)

# Restricao Orcamentaria
res_orc = lambda sal, lucro, laz, tempo: lucro + sal * (tempo - laz)

# Demanda por produtos
dem_por_prod = lambda exp, sal, tempo: exp * (1 / (4 * sal) + sal * tempo)

# Demanda por lazer
dem_laz = lambda exp, sal, tempo: (1 - exp) * (tempo + (1 / (4 * sal ** 2)))

# Oferta de trabalho
of_de_trab = lambda exp, sal, tempo: tempo - dem_laz(exp, sal, tempo)
 
###############################################################################
# Firma ----

# Funcao de producao
f_de_prod = lambda trab: np.sqrt(trab)

# Demanda por trabalho
dem_por_trab = lambda sal: 1 / (4 * sal ** 2)

# Oferta de produtos
of_de_prod = lambda sal: 1 / (2 * sal)

###############################################################################
# Equilibrio ----

excesso_de_dem = lambda exp, sal, tempo: ((2 - exp) / (4 * sal ** 2)) - exp * tempo

sal_med = lambda sal_s, sal_e: (np.sqrt(2) * sal_s * sal_e) /\
 np.sqrt(sal_s ** 2 + sal_e ** 2)

# Implementacao analitica do equilibrio
def eq_analitico(exp, tempo, tol = 5):
    
    # Formula para o salario de equilibrio encontrada
    # analiticamente
    sal_eq = np.sqrt((2 - exp) / (4 * tempo * exp))
    
    # Aplicando o salario de equilibrio nas demais funcoes
    of_trab = of_de_trab(exp, sal_eq, tempo)
    dem_trab = dem_por_trab(sal_eq)
    of_prod = of_de_prod(sal_eq)
    dem_prod = dem_por_prod(exp, sal_eq, tempo)
    laz = dem_laz(exp, sal_eq, tempo)
    
    # Garantindo que estamos no equilibrio
    assert round(of_trab, tol) == round(dem_trab, tol)
    assert round(of_prod, tol) == round(dem_prod, tol)
    
    # Lucro
    lucro = f_de_prod(of_trab) - sal_eq * of_trab
    
    return sal_eq, of_trab, of_prod, laz, lucro

# Funcao numerica para o equilibrio usando bissecao
def equilibrio(exp, tempo, sal_s, sal_e, tol = 5):
    
    # Garantindo que os valores de entrada
    # fazem sentido
    assert sal_s > 0 and sal_e > 0 and tempo > 0
    assert sal_s < sal_e
    
    while excesso_de_dem(exp, sal_e, tempo) > 0:
        
        sal_e = 2 * sal_e
    
    # Encontrando os primeiros pontos da funcao
    # excesso de demanda
    fs = excesso_de_dem(exp, sal_s, tempo)
    fe = excesso_de_dem(exp, sal_e, tempo)
    
    # Primeira media
    med = (fs + fe) / 2
    
    # Enquanto o valor absoluto da media 
    # arredondado para um erro for diferente de 0
    while round(abs(med), tol) != 0:
        
        # Se a media for positiva
        if med > 0:
            
            # Valor de inicio do salario sera o
            # valor do salario na media
            sal_s = sal_med(sal_s, sal_e)
        
        # Se a media for negativa
        elif med < 0:
            
            # Valor final do salario sera o valor
            # do salario na media
            sal_e = sal_med(sal_s, sal_e)
        
        # Atualiza valores
        fs = excesso_de_dem(exp, sal_s, tempo)
        fe = excesso_de_dem(exp, sal_e, tempo)
    
        med = (fs + fe) / 2
      
    # Aplicando as funcoes
    sal_eq = sal_med(sal_s, sal_e)
    of_trab = of_de_trab(exp, sal_eq, tempo)
    dem_trab = dem_por_trab(sal_eq)
    of_prod = of_de_prod(sal_eq)
    dem_prod = dem_por_prod(exp, sal_eq, tempo)
    laz = dem_laz(exp, sal_eq, tempo)
    
    # Garantindo que estamos no equilibrio
    # dada a tolerancia
    assert round(of_trab, tol) == round(dem_trab, tol)
    assert round(of_prod, tol) == round(dem_prod, tol)
    
    # Lucro
    lucro = f_de_prod(of_trab) - sal_eq * of_trab
    
    return sal_eq, of_trab, of_prod, laz, lucro
 
# Funcao numerica para o equilibrio usando bissecao
def equilibrio2(exp, tempo, sal_s, sal_e, tol = 5):
    
    # Garantindo que os valores de entrada
    # fazem sentido
    assert sal_s > 0 and sal_e > 0 and tempo > 0
    assert sal_s < sal_e
    
    while excesso_de_dem(exp, sal_e, tempo) > 0:
        
        sal_e = 2 * sal_e
    
    sal_m = (sal_s + sal_e) / 2
    
    # Enquanto o valor absoluto da media 
    # arredondado para um erro for diferente de 0
    while round(abs(excesso_de_dem(exp, sal_m, tempo)), tol) != 0:
        
        # Se a media for positiva
        if excesso_de_dem(exp, sal_m, tempo) > 0:
            
            # Valor de inicio do salario sera o
            # valor do salario na media
            sal_s = sal_m
        
        # Se a media for negativa
        elif excesso_de_dem(exp, sal_m, tempo) < 0:
            
            # Valor final do salario sera o valor
            # do salario na media
            sal_e = sal_m
        
        # Atualiza valores
        
        sal_m = (sal_s + sal_e) / 2
      
    # Aplicando as funcoes
    sal_eq = sal_med(sal_s, sal_e)
    of_trab = of_de_trab(exp, sal_eq, tempo)
    dem_trab = dem_por_trab(sal_eq)
    of_prod = of_de_prod(sal_eq)
    dem_prod = dem_por_prod(exp, sal_eq, tempo)
    laz = dem_laz(exp, sal_eq, tempo)
    
    # Garantindo que estamos no equilibrio
    # dada a tolerancia
    assert round(of_trab, tol) == round(dem_trab, tol)
    assert round(of_prod, tol) == round(dem_prod, tol)
    
    # Lucro
    lucro = f_de_prod(of_trab) - sal_eq * of_trab
    
    return sal_eq, of_trab, of_prod, laz, lucro   
###############################################################################
# Graficos ----

def cobb_douglas_inv(U, x2, exp):
    
    return (U / (x2 ** (1 - exp))) ** (1 / exp)

def graficos(exp, tempo, laz_s, laz_e, laz_t):
    
    # Criando vetores e matrizes
    laz_vec = np.linspace(laz_s, laz_e, laz_t)
    prod_vec = laz_vec.copy()
    X, Y = np.meshgrid(prod_vec, laz_vec)
    
    # Encontrando o equilibrio
    sal, trab, prod, laz, lucro = eq_analitico(exp, tempo)
    
    # Encontrando utilidade no equilibrio
    util = cobb_douglas(dem_por_prod(exp, sal, tempo),
                        dem_laz(exp, sal, tempo),
                        exp)
    
    # Gerando curvas de indiferenca
    U = cobb_douglas(X, Y, exp)
    
    # Plotando a restricao orcamentaria
    plt.plot(laz_vec, res_orc(sal, lucro, laz_vec, tempo),
             c = 'green',
             label = 'Restrição Orçamentária')
    
    # Plotando a funcao de producao
    plt.plot(laz_vec, f_de_prod(tempo - laz_vec),
             c = 'red',
             label = 'Função de Produção',
             linestyle = 'dotted')
    
    # Plotando curvas de indiferenca
    #plt.contour(Y, X, U,
                #levels = [util],
                #colors = 'purple',
                #linestyle = 'dashed')
    
    plt.plot(laz_vec, cobb_douglas_inv(util, laz_vec, exp),
             c = 'purple',
             label = 'Curva de Indiferença',
             linestyle = 'dashed')
    
    # Plotando o equilibrio
    plt.plot(laz, prod, 'go',
             c = 'black',
             label = 'Equilíbrio')
    
    plt.xlabel('Lazer')
    plt.ylabel('Consumo')
    plt.title('Equilíbrio Geral')
    plt.ylim([0, util + 1])
    plt.text(laz - 0.05, prod + 0.05, '({laz: .2f}, {prod: .2f})'.format(laz = laz, prod = prod))
    plt.legend(loc = 'best')
    plt.show()
    
graficos(0.7, 1, 0.0001, 1, 100)