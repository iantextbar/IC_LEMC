# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 12:19:47 2021

@author: Ian
"""

h_a = ['m_1', 'm_2', 'm_3', 's']
h_b = ['m_2', 'm_1', 'm_3', 's']
h_c = ['m_1', 's', 'm_3', 'm_2']
h_d = ['m_2', 'm_3', 's', 'm_1']

homens = ['h_a', 'h_b', 'h_c', 'h_d']

m_1 = ['h_b', 'h_c', 's', 'h_a', 'h_d']
m_2 = ['h_a', 's', 'h_c', 'h_b', 'h_d']
m_3 = ['h_d', 'h_a', 's', 'h_c', 'h_b']

casamentos = []

def propor_casamento_h(lista_homens):
    propostas_h = []
    cada_prop = []
    
    for homem in lista_homens:
        cada_prop.append(homem)
        cada_prop.append(eval(homem)[0])
        propostas_h.append(cada_prop)
        cada_prop = []
    
    return propostas_h

def organiza_prop(lista_propostas):
    
    prop_p_mulher_1 = []
    prop_p_mulher_2 = []
    prop_p_mulher_3 = []
    
    for prop in range(0, len(lista_propostas)):
        if lista_propostas[prop][1] == 'm_1':
            prop_p_mulher_1.append(lista_propostas[prop][0])
        elif lista_propostas[prop][1] == 'm_2':
            prop_p_mulher_2.append(lista_propostas[prop][0])
        else:
            prop_p_mulher_3.append(lista_propostas[prop][0])
    
    return prop_p_mulher_1, prop_p_mulher_2, prop_p_mulher_3
    

def mulher_avalia(propostas, preferencias):
    temp = ''
    while temp not in propostas or temp != 's':
        i = 0
        temp = preferencias[i]
        i += 1
    
    return temp
    

a = propor_casamento_h(homens)

p1, p2, p3 = organiza_prop(a)

escolha_m1 = mulher_avalia(p1, m_1)
