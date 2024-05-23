# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 19:25:40 2022

@author: Ian
"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Funcoes do programa anterior

def IS(x, cons_aut, p_marg_cons, G, T, c_inv, i_inv, is_inv = True):
    
    """
    x      = Caso is_inv = True serao valores do produto Y. Caso contrario
                      serao valores do juros r.
    is_inv = Se True (default) temos a IS invertida (r em funcao de Y)
    """
    
    if is_inv:
        ra = (x - x * p_marg_cons - cons_aut + p_marg_cons * T - i_inv - G)
        r = ra / c_inv
        return r

    else:
        mult = 1/(1 - p_marg_cons)
        gast_aut = cons_aut - (p_marg_cons * T) + (c_inv * x) + i_inv + G
        return mult * gast_aut

def eq_algebrico(cons_aut, p_marg_cons, G, T,
                 c_inv, i_inv, c_mon, i_mon):
    
    """
    Encontra o produto e o juros de equilibrio, de uma IS com investimento linear
    e uma LM linear. Utiliza equacoes encontradas algebricamente. 
    _________________________________
    cons_aut    = Consumo Autonomo
    p_marg_cons = Propensao Marginal a Consumir (0 <= x <= 1)
    G           = Gastos do Governo
    T           = Tributos
    c_inv       = Coeficiente Linear da Funcao de Investimento (x < 0)
    i_inv       = Intercepto da Funcao de Investimento
    c_mon       = Coeficiente Linear da LM (x > 0)
    i_mon       = Intercepto da LM
    """
    
    # Propensao Margina a Consumir positiva entre 0 e 1
    # Investimento negativamente relacionado ao juros
    # LM positivamente inclinada
    assert p_marg_cons >= 0 and p_marg_cons <= 1 and c_inv < 0 and c_mon > 0
    
    denom = (c_inv * c_mon) - 1 + p_marg_cons
    
    # Para evitar inf
    assert denom != 0
    
    # Montando a equacao
    ya = 1 / denom
    yb = ((p_marg_cons * T) - cons_aut - i_inv - G - (c_inv * i_mon))
    
    # Produto e juros de equilibrio
    Y = ya * yb
    r = c_mon * Y + i_mon
    
    return Y, r

###############################################################################
# Extensao

def definir_objetos():
    
    # Define os objetos utilizados nas demais funcoes
    
    cons_aut = 2
    p_marg_cons = 0.75
    G = 10
    T = 2
    c_inv = -3
    i_inv = 0
    of_mon = 6
    cy_dem_mon = 2
    cr_dem_mon = -2
    i_mon = 0
    
    assert p_marg_cons >= 0 and p_marg_cons <= 1
    assert c_inv < 0
    assert cy_dem_mon > 0
    assert cr_dem_mon < 0
    
    global pars1
    
    pars1 = [cons_aut, p_marg_cons, G, T, c_inv, i_inv, of_mon, \
             cy_dem_mon, cr_dem_mon, i_mon]
        
    cons_aut2 = 0.5
    p_marg_cons2 = 0.25
    G2 = 10
    T2 = 2
    c_inv2 = -1.5
    i_inv2 = 0
    of_mon2 = 6
    cy_dem_mon2 = 2.5
    cr_dem_mon2 = -0.3
    i_mon2 = 0
    
    assert p_marg_cons2 >= 0 and p_marg_cons2 <= 1
    assert c_inv2 < 0
    assert cy_dem_mon2 > 0
    assert cr_dem_mon2 < 0
    
    global pars2
    
    pars2 = [cons_aut2, p_marg_cons2, G2, T2, c_inv2, i_inv2, of_mon2, \
             cy_dem_mon2, cr_dem_mon2, i_mon2]

definir_objetos()

def pref_pela_liq(y, of_mon, cy_dem_mon, cr_dem_mon, i_mon):
    
    """
    Expressa o equilibrio no mercado monetario. Retorna a taxa de juros de
    equilibrio dada a oferta monetaria e a renda
    _________________________________
    y           = Renda
    of_mon      = Oferta Monetaria
    cy_dem_mon  = Coeficiente da renda na demanda monetaria
    cr_dem_mon  = Coeficiente do juros na demanda monetaria
    i_mon       = Intercepto da demanda monetaria
    """
    
    assert cr_dem_mon < 0 and cy_dem_mon > 0
    
    r = (of_mon - cy_dem_mon * y - i_mon) / cr_dem_mon
    
    return r

def eq_analitico_pref_pela_liq(pars):
    
    """
    Encontra a renda e a taxa de juros de equilibrio
    
    pars = cons_aut, p_marg_cons, G, T, c_inv, \
    i_inv, of_mon, cy_dem_mon, cr_dem_mon, i_mon
    _________________________________
    cons_aut    = Consumo Autonomo
    p_marg_cons = Propensao Marginal a Consumir (0 <= x <= 1)
    gast_gov    = Gastos do Governo
    tributos    = Tributos
    c_inv       = Coeficiente Linear da Funcao de Investimento (x < 0)
    i_inv       = Intercepto da Funcao de Investimento
    of_mon      = Oferta Monetaria
    cy_dem_mon  = Coeficiente da renda na demanda monetaria
    cr_dem_mon  = Coeficiente do juros na demanda monetaria
    i_mon       = Intercepto da demanda monetaria
    """
    
    cons_aut, p_marg_cons, G, T, c_inv, i_inv, \
    of_mon, cy_dem_mon, cr_dem_mon, i_mon = pars
    
    assert cr_dem_mon < 0 and cy_dem_mon > 0
    
    gas_aut = cons_aut - (p_marg_cons * T) + i_inv + G
    ya = (c_inv * (of_mon - i_mon)) + (cr_dem_mon * gas_aut)
    denom = (cr_dem_mon * (1 - p_marg_cons)) + (cy_dem_mon * c_inv)
    
    y = ya / denom
    r = pref_pela_liq(y, of_mon, cy_dem_mon, cr_dem_mon, i_mon)
    
    return y, r

def retorno_ao_eq_por_G(incremento, tol):
    
    """
    Retorna a renda de equilibrio anterior pelo aumento nos
    gastos do governo
    
    pars = cons_aut, p_marg_cons, G, T, c_inv, \
    i_inv, of_mon, cy_dem_mon, cr_dem_mon, i_mon
    _________________________________
    cons_aut    = Consumo Autonomo
    p_marg_cons = Propensao Marginal a Consumir (0 <= x <= 1)
    gast_gov    = Gastos do Governo
    tributos    = Tributos
    c_inv       = Coeficiente Linear da Funcao de Investimento (x < 0)
    i_inv       = Intercepto da Funcao de Investimento
    of_mon      = Oferta Monetaria
    cy_dem_mon  = Coeficiente da renda na demanda monetaria
    cr_dem_mon  = Coeficiente do juros na demanda monetaria
    i_mon       = Intercepto da demanda monetaria
    """
    
    # Calcula equilibrios
    Y1, r1 = eq_analitico_pref_pela_liq(pars1)
    
    Y2, r2 = eq_analitico_pref_pela_liq(pars2)
    
    y_eq_vec = [Y1, Y2]
    r_eq_vec = [r1, r2]
    t = 0
    
    # Checa se os equilibrios sao iguais
    if Y1 == Y2 and r1 == r2:
        
        print('Mesmo equilibrio')
        return 0
    
    # Caso a renda tenha aumentado com
    # os novos parametros
    elif Y1 < Y2:
        
        print('A renda aumentou!')
        return 0
    
    else:
        
        # Enquanto as rendas forem diferentes
        while round(abs(Y1 - Y2), tol) != 0:
                
            # Aumenta o gasto do governo - pars2[2] sao os gastos
            # do governo
            pars2[2] += incremento
                
            # Calcula o novo equilibrio
            Y2, r2 = eq_analitico_pref_pela_liq(pars2)
            t += 1
                
            y_eq_vec.append(Y2)
            r_eq_vec.append(r2)
                
    return pars2, y_eq_vec, r_eq_vec, t
                 
def retorno_ao_eq_por_T(incremento, tol):
    
    """
    Retorna a renda de equilibrio anterior pela diminuicao nos impostos
    
    pars = cons_aut, p_marg_cons, G, T, c_inv, \
    i_inv, of_mon, cy_dem_mon, cr_dem_mon, i_mon
    _________________________________
    cons_aut    = Consumo Autonomo
    p_marg_cons = Propensao Marginal a Consumir (0 <= x <= 1)
    gast_gov    = Gastos do Governo
    tributos    = Tributos
    c_inv       = Coeficiente Linear da Funcao de Investimento (x < 0)
    i_inv       = Intercepto da Funcao de Investimento
    of_mon      = Oferta Monetaria
    cy_dem_mon  = Coeficiente da renda na demanda monetaria
    cr_dem_mon  = Coeficiente do juros na demanda monetaria
    i_mon       = Intercepto da demanda monetaria
    """
    
    # Calcula equilibrios
    Y1, r1 = eq_analitico_pref_pela_liq(pars1)
    
    Y2, r2 = eq_analitico_pref_pela_liq(pars2)
    
    y_eq_vec = [Y1, Y2]
    r_eq_vec = [r1, r2]
    t = 0
    
    # Checa se os equilibrios sao iguais
    if Y1 == Y2 and r1 == r2:
        
        print('Mesmo equilibrio')
        return 0
    
    # Caso a renda tenha aumentado com
    # os novos parametros
    elif Y1 < Y2:
        
        print('A renda aumentou!')
        return 0
    
    else:
        
        # Enquanto as rendas forem diferentes
        while round(abs(Y1 - Y2), tol) != 0:
            
            # Diminui os impostos - pars2[3] sao os impostos
            pars2[3] -= incremento
            
            # Calcula o novo equilibrio
            Y2, r2 = eq_analitico_pref_pela_liq(pars2)
            
            t += 1
                
            y_eq_vec.append(Y2)
            r_eq_vec.append(r2)
                
    return pars2, y_eq_vec, r_eq_vec, t


def retorno_ao_eq_orc_eq(incremento, tol):
    
    #assert G2 == T2
    
    # Calcula equilibrios
    Y1, r1 = eq_analitico_pref_pela_liq(pars1)
    
    Y2, r2 = eq_analitico_pref_pela_liq(pars2)
    
    y_eq_vec = [Y1, Y2]
    r_eq_vec = [r1, r2]
    t = 0
    
    # Checa se os equilibrios sao iguais
    if Y1 == Y2 and r1 == r2:
        
        print('Mesmo equilibrio')
        return 0
    
    # Caso a renda tenha aumentado com
    # os novos parametros
    elif Y1 < Y2:
        
        print('A renda aumentou!')
        return 0
    
    else:
        
        while round(abs(Y1 - Y2), tol) != 0:
            
            pars2[2] += incremento
            pars2[3] = pars2[2]
            
            # Calcula o novo equilibrio
            Y2, r2 = eq_analitico_pref_pela_liq(pars2)
            
            t += 1
                
            y_eq_vec.append(Y2)
            r_eq_vec.append(r2)
            
    return pars2, y_eq_vec, r_eq_vec, t

def retorno_ao_eq_pol_mon(incremento, tol):
    
    """
    Retorna a renda de equilibrio anterior pelo aumento da oferta
    monetaria
    
    pars = cons_aut, p_marg_cons, G, T, c_inv, \
    i_inv, of_mon, cy_dem_mon, cr_dem_mon, i_mon
    _________________________________
    cons_aut    = Consumo Autonomo
    p_marg_cons = Propensao Marginal a Consumir (0 <= x <= 1)
    gast_gov    = Gastos do Governo
    tributos    = Tributos
    c_inv       = Coeficiente Linear da Funcao de Investimento (x < 0)
    i_inv       = Intercepto da Funcao de Investimento
    of_mon      = Oferta Monetaria
    cy_dem_mon  = Coeficiente da renda na demanda monetaria
    cr_dem_mon  = Coeficiente do juros na demanda monetaria
    i_mon       = Intercepto da demanda monetaria
    """
    
    # Calcula equilibrios
    Y1, r1 = eq_analitico_pref_pela_liq(pars1)
    
    Y2, r2 = eq_analitico_pref_pela_liq(pars2)
    
    y_eq_vec = [Y1, Y2]
    r_eq_vec = [r1, r2]
    t = 0

    # Checa se os equilibrios sao iguais
    if Y1 == Y2 and r1 == r2:
        
        print('Mesmo equilibrio')
        return 0
    
    # Caso a renda tenha aumentado com
    # os novos parametros
    elif Y1 < Y2:
        
        print('A renda aumentou!')
        return 0
    
    else:
        while round(abs(Y1 - Y2), tol) != 0:
            
            # Aumento da oferta monetaria
            pars2[6] += incremento
            
            # Calcula o novo equilibrio
            Y2, r2 = eq_analitico_pref_pela_liq(pars2)
            
            t += 1
                
            y_eq_vec.append(Y2)
            r_eq_vec.append(r2)
            
            if round(abs(r2), tol) == 0:
                
                print('Armadilha da liquidez')
                return pars2, y_eq_vec, r_eq_vec, t

    return pars2, y_eq_vec, r_eq_vec, t

eq1 = eq_analitico_pref_pela_liq(pars2)
eq2 = eq_analitico_pref_pela_liq(pars1)

exp1 = retorno_ao_eq_por_G(0.01, 4)

definir_objetos()

exp2 = retorno_ao_eq_por_T(0.01, 4)

definir_objetos()

#exp3 = retorno_ao_eq_orc_eq(0.01, 4)
exp4 = retorno_ao_eq_pol_mon(0.01, 4)

definir_objetos()

###############################################################################
# Graficos

y_vec = np.linspace(0, 9, 100)

# Equilibrio inicial
plt.plot(y_vec, pref_pela_liq(y_vec, pars1[6], pars1[7], pars1[8], pars1[9]), c = 'purple', label = 'LM')
plt.plot(y_vec, IS(y_vec, pars1[0], pars1[1], pars1[2], pars1[3], pars1[4], pars1[5]), c = 'red', label = 'IS')
plt.plot(eq2[0], eq2[1], 'go', c = 'black', label = 'Equilibrio Inicial')
plt.title('Equilibrio Inicial')
plt.xlabel('Y')
plt.ylabel('r')
plt.legend()
plt.text(eq2[0] * 0.9, eq2[1] * 1.1,
         '({y}, {r})'.format(y = eq2[0], r = eq2[1]))
plt.show()

# Novo equilibrio
plt.plot(y_vec, pref_pela_liq(y_vec, pars1[6], pars1[7], pars1[8], pars1[9]), c = 'purple', label = 'LM')
plt.plot(y_vec, IS(y_vec, pars1[0], pars1[1], pars1[2], pars1[3], pars1[4], pars1[5]), c = 'red', label = 'IS')
plt.plot(eq2[0], eq2[1], 'go', c = 'black', label = 'Equilibrio Inicial')
plt.plot(y_vec, pref_pela_liq(y_vec, pars2[6], pars2[7], pars2[8], pars2[9]), c = 'blue', label = 'LM choque')
plt.plot(y_vec, IS(y_vec, pars2[0], pars2[1], pars2[2], pars2[3], pars2[4], pars2[5]), c = 'green', label = 'IS choque')
plt.plot(eq1[0], eq1[1], 'go', c = 'orange', label = 'Equilibrio Choque')
plt.title('Novo equilibrio')
plt.xlabel('Y')
plt.ylabel('r')
plt.legend()
plt.text(eq2[0] * 0.9, eq2[1] * 1.1,
         '({y}, {r})'.format(y = eq2[0], r = eq2[1]))
plt.text(eq1[0] * 0.9, eq1[1] * 1.1,
         '({y:.2f}, {r:.2f})'.format(y = eq1[0], r = eq1[1]))
plt.show()

# Retorno ao equilibrio por G
plt.plot(y_vec, pref_pela_liq(y_vec, pars2[6], pars2[7], pars2[8], pars2[9]), c = 'blue', label = 'LM')
plt.plot(y_vec, IS(y_vec, pars2[0], pars2[1], pars2[2], pars2[3], pars2[4], pars2[5]), c = 'green', label = 'IS')
plt.plot(y_vec, IS(y_vec, pars2[0], pars2[1], exp1[0][2], pars2[3], pars2[4], pars2[5]), c = 'orange', label = 'IS aumento G')
plt.plot(eq1[0], eq1[1], 'go', c = 'black', label = 'Novo equilibrio')
plt.plot(exp1[1][-1], exp1[2][-1], 'go', c = 'orange', label = 'Volta ao equilibrio')
plt.title('Retorno ao equilibrio por G')
plt.xlabel('Y')
plt.ylabel('r')
plt.legend()
plt.text(eq1[0] * 0.9, eq1[1] * 1.1,
         '({y:.2f}, {r:.2f})'.format(y = eq1[0], r = eq1[1]))
plt.text(exp1[1][-1] * 0.9, exp1[2][-1] * 1.1,
         '({y:.2f}, {r:.2f})'.format(y = exp1[1][-1], r = exp1[2][-1]))
plt.show()

# Retorno ao equilibrio por T
plt.plot(y_vec, pref_pela_liq(y_vec, pars2[6], pars2[7], pars2[8], pars2[9]), c = 'blue', label = 'LM')
plt.plot(y_vec, IS(y_vec, pars2[0], pars2[1], pars2[2], pars2[3], pars2[4], pars2[5]), c = 'green', label = 'IS')
plt.plot(y_vec, IS(y_vec, pars2[0], pars2[1], pars2[2], exp2[0][3], pars2[4], pars2[5]), c = 'fuchsia', label = 'IS redução T')
plt.plot(eq1[0], eq1[1], 'go', c = 'black', label = 'Novo equilibrio')
plt.plot(exp2[1][-1], exp2[2][-1], 'go', c = 'orange', label = 'Volta ao equilibrio')
plt.title('Retorno ao equilibrio por T')
plt.xlabel('Y')
plt.ylabel('r')
plt.legend()
plt.text(eq1[0] * 0.9, eq1[1] * 1.1,
         '({y:.2f}, {r:.2f})'.format(y = eq1[0], r = eq1[1]))
plt.text(exp2[1][-1] * 0.9, exp2[2][-1] * 1.1,
         '({y:.2f}, {r:.2f})'.format(y = exp2[1][-1], r = exp2[2][-1]))
plt.show()

# Retorno ao equilibrio por politica monetaria
plt.plot(y_vec, pref_pela_liq(y_vec, pars2[6], pars2[7], pars2[8], pars2[9]), c = 'blue', label = 'LM')
plt.plot(y_vec, pref_pela_liq(y_vec, exp4[0][6], pars2[7], pars2[8], pars2[9]), c = 'gold', label = 'LM aumento oferta monetária')
plt.plot(y_vec, IS(y_vec, pars2[0], pars2[1], pars2[2], pars2[3], pars2[4], pars2[5]), c = 'green', label = 'IS')
plt.plot(eq1[0], eq1[1], 'go', c = 'black', label = 'Novo equilibrio')
plt.plot(exp4[1][-1], exp4[2][-1], 'go', c = 'orange', label = 'Volta ao equilibrio')
plt.title('Retorno ao equilibrio pela Política Monetária Expansionista')
plt.xlabel('Y')
plt.ylabel('r')
plt.legend()
plt.text(eq1[0] * 0.9, eq1[1] * 1.1,
         '({y:.2f}, {r:.2f})'.format(y = eq1[0], r = eq1[1]))
plt.text(exp4[1][-1] * 0.9, exp4[2][-1] * 1.1,
         '({y:.2f}, {r:.2f})'.format(y = exp4[1][-1], r = exp4[2][-1]))
plt.show()
###############################################################################
# Apendice da Extensao

def retorno_ao_eq_por_G_2(pars1, pars2, incremento, tol):
    
    # Recebe parametros
    cons_aut1, p_marg_cons1, G1, T1, c_inv1, i_inv1, c_mon1, i_mon1 = pars1
    cons_aut2, p_marg_cons2, G2, T2, c_inv2, i_inv2, c_mon2, i_mon2 = pars2
    
    # Calcula equilibrios
    Y1, r1 = eq_algebrico(cons_aut1, p_marg_cons1, G1, T1, c_inv1,
                          i_inv1, c_mon1, i_mon1)
    
    Y2, r2 = eq_algebrico(cons_aut2, p_marg_cons2, G2, T2, c_inv2,
                          i_inv2, c_mon2, i_mon2)
    
    y_eq_vec = [Y1, Y2]
    r_eq_vec = [r1, r2]
    t = 0
    
    # Checa se os equilibrios sao iguais
    if Y1 == Y2 and r1 == r2:
        
        print('Mesmo equilibrio')
        return 0
    
    # Caso a renda tenha aumentado com
    # os novos parametros
    elif Y1 < Y2:
        
        print('A renda aumentou!')
        return 0
    
    else:
        
        # Enquanto as rendas forem diferentes
        while round(abs(Y1 - Y2), tol) != 0:
                
            # Aumenta o gasto do governo
            G2 += incremento
                
            # Calcula o novo equilibrio
            Y2, r2 = eq_algebrico(cons_aut2, p_marg_cons2, G2, 
                                  T2, c_inv2, i_inv2, c_mon2, i_mon2)
            t += 1
                
            y_eq_vec.append(Y2)
            r_eq_vec.append(r2)
                
    return G2, T2, y_eq_vec, r_eq_vec, t
                  
def retorno_ao_eq_por_T_2(pars1, pars2, incremento, tol):
    
    # Recebe parametros
    cons_aut1, p_marg_cons1, G1, T1, c_inv1, i_inv1, c_mon1, i_mon1 = pars1
    cons_aut2, p_marg_cons2, G2, T2, c_inv2, i_inv2, c_mon2, i_mon2 = pars2
    
    # Calcula equilibrios
    Y1, r1 = eq_algebrico(cons_aut1, p_marg_cons1, G1, T1, c_inv1,
                          i_inv1, c_mon1, i_mon1)
    
    Y2, r2 = eq_algebrico(cons_aut2, p_marg_cons2, G2, T2, c_inv2,
                          i_inv2, c_mon2, i_mon2)
    
    y_eq_vec = [Y1, Y2]
    r_eq_vec = [r1, r2]
    t = 0
    
    # Checa se os equilibrios sao iguais
    if Y1 == Y2 and r1 == r2:
        
        print('Mesmo equilibrio')
        return 0
    
    # Caso a renda tenha aumentado com
    # os novos parametros
    elif Y1 < Y2:
        
        print('A renda aumentou!')
        return 0
    
    else:
        
        # Enquanto as rendas forem diferentes
        while round(abs(Y1 - Y2), tol) != 0:
            
            # Diminui os impostos
            T2 -= incremento
            
            # Calcula o novo equilibrio
            Y2, r2 = eq_algebrico(cons_aut2, p_marg_cons2, G2,
                                  T2, c_inv2, i_inv2, c_mon2, i_mon2)
            
            t += 1
                
            y_eq_vec.append(Y2)
            r_eq_vec.append(r2)
                
    return G2, T2, y_eq_vec, r_eq_vec, t


def retorno_ao_eq_orc_eq_2(pars1, pars2, incremento, tol):
    
    # Recebe parametros
    cons_aut1, p_marg_cons1, G1, T1, c_inv1, i_inv1, c_mon1, i_mon1 = pars1
    cons_aut2, p_marg_cons2, G2, T2, c_inv2, i_inv2, c_mon2, i_mon2 = pars2
    
    #assert G2 == T2
    
    # Calcula equilibrios
    Y1, r1 = eq_algebrico(cons_aut1, p_marg_cons1, G1, T1, c_inv1,
                          i_inv1, c_mon1, i_mon1)
    
    Y2, r2 = eq_algebrico(cons_aut2, p_marg_cons2, G2, T2, c_inv2,
                          i_inv2, c_mon2, i_mon2)
    
    y_eq_vec = [Y1, Y2]
    r_eq_vec = [r1, r2]
    t = 0
    
    # Checa se os equilibrios sao iguais
    if Y1 == Y2 and r1 == r2:
        
        print('Mesmo equilibrio')
        return 0
    
    # Caso a renda tenha aumentado com
    # os novos parametros
    elif Y1 < Y2:
        
        print('A renda aumentou!')
        return 0
    
    else:
        
        while round(abs(Y1 - Y2), tol) != 0:
            
            G2 += incremento
            T2 = G2
            
            # Calcula o novo equilibrio
            Y2, r2 = eq_algebrico(cons_aut2, p_marg_cons2, G2,
                                  T2, c_inv2, i_inv2, c_mon2, i_mon2)
            
            t += 1
                
            y_eq_vec.append(Y2)
            r_eq_vec.append(r2)
            
    return G2, T2, y_eq_vec, r_eq_vec, t