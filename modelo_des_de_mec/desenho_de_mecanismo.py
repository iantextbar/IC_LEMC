# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:19:08 2022

@author: Ian

Desenho de Mecanismo
"""
import numpy as np
from itertools import product

theta1 = [1]
theta2 = [1, 2]
thetas = [[1], [1, 2]]
X = [1, 2, 3]

def dominio(lista_thetas):
    
    """
    Funcao cria o dominio de f
    """
    
    # Aplica o produto cartesiano entre os conjuntos
    # dos tipos de cada agente
    prod_cart = list(product(*lista_thetas))
    
    # Transforma a lista de tuplas em uma matriz
    dom = [[j for j in prod_cart[i]] for i in range(len(prod_cart))]

    return np.array(dom)

def social_choice_functions(dominio, contradominio):
    
    """
    Cria todas as possibilidades de funcoes de escolha social
    """
    
    # Para cada elemento no dominio cria uma copia
    # da lista de escolhas publicas disponiveis
    aux = [contradominio for i in range(len(dominio))]
    
    # Faz o produto cartesiano entre cada uma das copias
    # da lista de escolhas publicas
    prod_cart = list(product(*aux))
    
    # Transforma a lista de tuplas em uma matriz
    scf = [[j for j in prod_cart[i]] for i in range(len(prod_cart))]

    return np.array(scf)    

def f_u2(choice, theta):
    
    """
    Funcao utilidade da pessoa 2
    """
    
    # Caso a tecnologia seja b2 e a escolha seja z
    if theta == 2 and choice == 3:
        
        return 25
    
    # Caso contrario
    else:
        
        if choice == 1:
            
            return 0
        
        elif choice == 2:
            
            return 50
        
        else:
            
            return 100
    
def utilidades(scf, f_util, thetas):
    
    """
    Recebe todas as funcoes de escolha social, a funcao utilidade
    de um agente e o conjunto thetas de tipos daquele agente. Retorna
    um tensor com a utilidade do agente para cada tipo e cada funcao
    """
    
    # Cria lista de utilidades - ha len(scf) listas, representando
    # cada funcao de escolha social possivel. Dentro de cada uma 
    # dessas listas ha len(thetas) representando cada tipo de agente
    u = [[[] for j in range(len(thetas))] for i in range(len(scf))]
    
    # Para cada funcao de escolha social f
    for f in range(len(scf)):
        
        # Para cada tipo t do agente
        for t in range(len(thetas)):
            
            # Para cada escolha social feita, apensa a
            # utilidade do agente caso ele for do tipo t
            for c in scf[f]:
                
                u[f][t].append(f_util(c, thetas[t]))
    
    return u
        
    
def implementavel_facil(scf, thetas, utilidades):
    
    """
    Encontra as funcoes de escolha social implementaveis
    """
    
    # Cria a mascara preenchida com True
    implementavel = [True for i in range(len(scf))]
    
    # Para cada funcao de escolha social
    for f in range(len(utilidades)):
        
        # Para cada tipo do agente 2
        for t in range(len(thetas)):
            
            # Se as utilidades de cada tipo forem iguais
            # passa para a proxima iteracao - sera uma
            # implementacao valida para aquele tipo
            if utilidades[f][t][0] == utilidades[f][t][1]:
                
                break
            
            # Caso o maior payoff seja mentir, ou seja
            # seja dizer que voce eh de outro tipo que
            # nao o seu verdadeiro, a funcao nao eh 
            # implementavel
            elif np.argmax(utilidades[f][t]) != t:
                
                implementavel[f] = False
    
    # Aplica a mascara no conjunto de todas as
    # funcoes de escolha social
    impl = scf[implementavel]
          
    return impl

dom = dominio(thetas)
scf = social_choice_functions(dom, X)
util = utilidades(scf, f_u2, theta2)
impl = implementavel_facil(scf, theta2, util)