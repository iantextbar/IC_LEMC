# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:26:14 2021

@author: Ian
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set

###############################################################################
# Funcoes Basicas

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

prod, juro = eq_algebrico(2, 0.75, 10, 2, -3, 0, 2, 1)
p, j = eq_algebrico(1, 0.5, 3, 4, -1, 0, 3, -2)
p_red_t, j_red_t = eq_algebrico(2, 0.75, 10, 0.5, -3, 0, 2, 1)
p_au_g, j_au_g = eq_algebrico(2, 0.75, 14, 2, -3, 0, 2, 1)
p_mon_con, j_mon_con = eq_algebrico(2, 0.75, 10, 2, -3, 0, 2, 3)
p_mon_exp, j_mon_exp = eq_algebrico(2, 0.75, 10, 2, -3, 0, 2, 0.25)

def equilibrio(cons_aut, p_marg_cons, gast_gov, tributos,
               c_inv, i_inv, c_mon, i_mon,
               grid_s, grid_e, grid_tam = 1000):
    
    """
    Encontra o produto e o juros de equilibrio, de uma IS com investimento linear
    e uma LM linear. A tecnica do ponto fixo. 
    _________________________________
    cons_aut    = Consumo Autonomo
    p_marg_cons = Propensao Marginal a Consumir (0 <= x <= 1)
    gast_gov    = Gastos do Governo
    tributos    = Tributos
    c_inv       = Coeficiente Linear da Funcao de Investimento (x < 0)
    i_inv       = Intercepto da Funcao de Investimento
    c_mon       = Coeficiente Linear da LM (x > 0)
    i_mon       = Intercepto da LM
    grid_s      = Inicio do grid
    grid_e      = Final do grid
    grid_tam    = Tamanho do grid
    """
    
    # Declarando variaveis globais
    # que serao usadas em outras funcoes
    global c0, c1, G, T, coef_i, inter_i, coef_lm, inter_lm
    
    # Atribuindo valores as variaveis globais
    # de acordo com os parametros
    c0, c1, G, T, coef_i, inter_i, coef_lm, inter_lm = [cons_aut, p_marg_cons,
                                                        gast_gov, tributos,
                                                        c_inv, i_inv,
                                                        c_mon, i_mon]
    
    # Propensao Margina a Consumir positiva entre 0 e 1
    # Investimento negativamente relacionado ao juros
    # LM positivamente inclinada
    assert c1 >= 0 and c1 <= 1 and coef_i < 0 and coef_lm > 0
    
    # Criando o grid
    grid_y = np.linspace(grid_s, grid_e, grid_tam)
    
    diff = []
    
    # Preenche diff com o valor absoluto da diferenca entre
    # a IS e a LM
    for val in grid_y:
        diff.append(abs(IS(val) - LM(val)))
    
    diff = np.array(diff)
    
    # Encontra os valores de interesse
    prod_eq = grid_y[np.argmin(diff)]
    juros_eq = IS(prod_eq)
    
    return prod_eq, juros_eq
    


def IS(x, is_inv = True):
    
    """
    x      = Caso is_inv = True serao valores do produto Y. Caso contrario
                      serao valores do juros r.
    is_inv = Se True (default) temos a IS invertida (r em funcao de Y)
    """
    
    if is_inv:
        return (x - x * c1 - c0 + c1 * T - inter_i - G)/coef_i

    else:
        mult = 1/(1 - c1)
        gast_aut = c0 - (c1 * T) + (coef_i * x) + inter_i + G
        return mult * gast_aut

def LM(x, lm_inv = False):
    
    """
    x      = Caso lm_inv = False serao valores do produto Y. Caso contrario
                      serao valores do juros r.
    lm_inv = Se False (default) temos a LM invertida (r em funcao de Y)
    """
    
    if lm_inv:
        return (x - inter_lm) / coef_lm
    else:
        return coef_lm * x + inter_lm
    
prod2, juro2 = equilibrio(2, 0.75, 10, 2, -3, 0, 2, 1, 0, 4)
p2, j2 = equilibrio(1, 0.5, 3, 4, -1, 0, 3, -2, 0, 2)

def eq_linalg(cons_aut, p_marg_cons, G_gov, T_gov,
               c_inv, i_inv, c_mon, i_mon):
    
    
    vet_const_1 = cons_aut - (p_marg_cons * T_gov) + i_inv + G_gov
    vet_const = np.array([[vet_const_1], [i_mon]])
    
    mat_coef = np.array([[1 - p_marg_cons, -c_inv],[-c_mon, 1]])
    inv_mat = np.linalg.inv(mat_coef)
    res = np.dot(inv_mat, vet_const)
    
    y, r = res
    
    return y, r

prod_mat, j_mat = eq_linalg(2, 0.75, 10, 2, -3, 0, 2, 1)

###############################################################################
# Extensao

def definir_objetos():
    
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
    
    pars1 = [cons_aut, p_marg_cons, G, T, c_inv, i_inv, of_mon, cy_dem_mon, cr_dem_mon, i_mon]
        
    cons_aut2 = 1.75
    p_marg_cons2 = 0.5
    G2 = 10
    T2 = 2
    c_inv2 = -4
    i_inv2 = 1
    of_mon2 = 6
    cy_dem_mon2 = 1.1
    cr_dem_mon2 = -2.3
    i_mon2 = 0
    
    assert p_marg_cons2 >= 0 and p_marg_cons2 <= 1
    assert c_inv2 < 0
    assert cy_dem_mon2 > 0
    assert cr_dem_mon2 < 0
    
    global pars2
    
    pars2 = [cons_aut2, p_marg_cons2, G2, T2, c_inv2, i_inv2, of_mon2, cy_dem_mon2, cr_dem_mon2, i_mon2]

definir_objetos()

def pref_pela_liq(y, of_mon, cy_dem_mon, cr_dem_mon, i_mon):
    
    assert cr_dem_mon < 0 and cy_dem_mon > 0
    
    r = (of_mon - cy_dem_mon * y - i_mon) / cr_dem_mon
    
    return r

def eq_analitico_pref_pela_liq(pars):
    
    cons_aut, p_marg_cons, G, T, c_inv, i_inv, of_mon, cy_dem_mon, cr_dem_mon, i_mon = pars
    
    assert cr_dem_mon < 0 and cy_dem_mon > 0
    
    gas_aut = cons_aut - (p_marg_cons * T) + i_inv + G
    ya = (c_inv * (of_mon - i_mon)) + (cr_dem_mon * gas_aut)
    denom = (cr_dem_mon * (1 - p_marg_cons)) + (cy_dem_mon * c_inv)
    
    y = ya / denom
    r = pref_pela_liq(y, of_mon, cy_dem_mon, cr_dem_mon, i_mon)
    
    return y, r

a = eq_analitico_pref_pela_liq(pars2)
b = eq_analitico_pref_pela_liq(pars1)

def retorno_ao_eq_por_G(incremento, tol):
    
    """
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
                
            # Aumenta o gasto do governo
            pars2[2] += incremento
                
            # Calcula o novo equilibrio
            Y2, r2 = eq_analitico_pref_pela_liq(pars2)
            t += 1
                
            y_eq_vec.append(Y2)
            r_eq_vec.append(r2)
                
    return pars2[2], pars2[3], y_eq_vec, r_eq_vec, t
                 
exp1 = retorno_ao_eq_por_G(0.01, 4)
    
def retorno_ao_eq_por_T2(pars1, pars2, incremento, tol):
    
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

exp2 = retorno_ao_eq_por_T([2, 0.75, 10, 2, -3, 0, 2, 1],
                           [1.75, 0.5, 10, 2, -4, 1, 3, 1],
                            0.01, 4)

def retorno_ao_eq_orc_eq2(pars1, pars2, incremento, tol):
    
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

exp3 = retorno_ao_eq_orc_eq([2, 0.75, 10, 2, -3, 0, 2, 1],
                           [1.75, 0.5, 10, 2, -4, 1, 3, 1],
                            0.01, 4)

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


###############################################################################
# Graficos

d = np.arange(0, 3, 0.0001)

# Apenas a curva IS
plt.plot(d, (-0.25 * d + 10.5)/(3), c = 'darkblue', label = 'IS: $Y = 2 + 0.75 \cdot (Y - 2) + -3 \cdot r + 10$')
plt.legend(loc = 'best', fontsize = 10)
plt.title('A Curva IS', fontsize = 20, fontweight = 700)
plt.xlabel('Y', fontsize = 15, fontweight = 600)
plt.ylabel('r', fontsize = 15, fontweight = 600, rotation = 0)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.show()

# A curva LM
plt.plot(d, (2 * d + 1), c = 'darkred', label = 'LM: $Y = {1/2} \cdot r - 1$')
plt.legend(loc = 'best', fontsize = 10)
plt.title('A Curva LM', fontsize = 20, fontweight = 700)
plt.xlabel('Y', fontsize = 15, fontweight = 600)
plt.ylabel('r', fontsize = 15, fontweight = 600, rotation = 0)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.show()

# O equilibrio

plt.plot(d, (2 * d + 1), c = 'darkred', label = 'LM: $Y = {1/2} \cdot r - 1$')
plt.plot(d, (-0.25 * d + 10.5)/(3), c = 'darkblue', label = 'IS: $Y = 2 + 0.75 \cdot (Y - 2) + -3 \cdot r + 10$')
plt.plot(prod, juro, 'go', c = 'black', label = 'Produto de Equilíbrio')
plt.text(prod - .1, juro + 0.3, '1.2')
plt.legend(loc = 'best', fontsize = 10)
plt.title('O Equiíbrio', fontsize = 20, fontweight = 700)
plt.xlabel('Y', fontsize = 15, fontweight = 600)
plt.ylabel('r', fontsize = 15, fontweight = 600, rotation = 0)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)
plt.show()

# O equilibrio e politica fiscal

fig, axs = plt.subplots(3, 1, sharex = True, figsize = (12, 8))
axs[0].plot(d, (2 * d + 1), c = 'darkred', label = 'LM: $Y = {1/2} \cdot r - 1$')
axs[0].plot(d, (-0.25 * d + 10.5)/(3), c = 'darkblue', label = 'IS: $Y = 2 + 0.75 \cdot (Y - 2) + -3 \cdot r + 10$')
axs[0].plot(prod, juro, 'go', c = 'black')
axs[0].set_title('O Equilíbrio Inicial', fontsize = 20, fontweight = 700)
axs[0].legend(loc = 'upper center',
              bbox_to_anchor = (1.2, 0.5),
              fancybox = True,
              fontsize = 10)
axs[0].text(prod - .1, juro + 0.3, '1.2', fontsize = 15, fontweight = 600)

axs[1].plot(d, (2 * d + 1), c = 'darkred', label = 'LM: $Y = {1/2} \cdot r - 1$')
axs[1].plot(d, (-0.25 * d + 11.625)/(3), c = 'darkblue', label = 'IS: $Y = 2 + 0.75 \cdot (Y - 0.5) + -3 \cdot r + 10$')
axs[1].plot(p_red_t, j_red_t, 'go', c = 'black')
axs[1].set_title('Redução de T para 0.5', fontsize = 20, fontweight = 700)
axs[1].legend(loc = 'upper center',
              bbox_to_anchor = (1.2, 0.5),
              fancybox = True,
              fontsize = 10)
axs[1].text(p_red_t - .1, j_red_t + 0.3, '1.38', fontsize = 15, fontweight = 600)
axs[1].set_ylabel('r', fontsize = 20, fontweight = 600, rotation = 0)


axs[2].plot(d, (2 * d + 1), c = 'darkred', label = 'LM: $Y = {1/2} \cdot r - 1$')
axs[2].plot(d, (-0.25 * d + 14.50)/(3), c = 'darkblue', label = 'IS: $Y = 2 + 0.75 \cdot (Y - 2) + -3 \cdot r + 14$')
axs[2].plot(p_au_g, j_au_g, 'go', c = 'black')
axs[2].set_title('Aumento de G para 14', fontsize = 20, fontweight = 700)
axs[2].legend(loc = 'upper center',
              bbox_to_anchor = (1.2, 0.5),
              fancybox = True,
              fontsize = 10)
axs[2].text(p_au_g - .1, j_au_g + 0.3, '1.84', fontsize = 15, fontweight = 600)
axs[2].set_xlabel('Y', fontsize = 20, fontweight = 600)
plt.tight_layout()
plt.show()

# O equilibrio e a politica monetaria
fig, axs = plt.subplots(3, 1, sharex = True, sharey = True, figsize = (12, 8))
axs[0].plot(d, (2 * d + 1), c = 'darkred', label = 'LM: $Y = {1/2} \cdot r - 1$')
axs[0].plot(d, (-0.25 * d + 10.5)/(3), c = 'darkblue', label = 'IS: $Y = 2 + 0.75 \cdot (Y - 2) + -3 \cdot r + 10$')
axs[0].plot(prod, juro, 'go', c = 'black')
axs[0].set_title('O Equilíbrio Inicial', fontsize = 20, fontweight = 700)
axs[0].legend(loc = 'upper center',
              bbox_to_anchor = (1.2, 0.5),
              fancybox = True,
              fontsize = 10)
axs[0].text(prod - .1, juro + 0.3, '1.2', fontsize = 15, fontweight = 600)

axs[1].plot(d, (2 * d + 3), c = 'darkred', label = 'LM: $Y = {1/2} \cdot r - 3$')
axs[1].plot(d, (-0.25 * d + 10.5)/(3), c = 'darkblue', label = 'IS: $Y = 2 + 0.75 \cdot (Y - 2) + -3 \cdot r + 10$')
axs[1].plot(p_mon_con, j_mon_con, 'go', c = 'black')
axs[1].set_title('Política Monetária Contracionista', fontsize = 20, fontweight = 700)
axs[1].legend(loc = 'upper center',
              bbox_to_anchor = (1.2, 0.5),
              fancybox = True,
              fontsize = 10)
axs[1].text(p_mon_con - .1, j_mon_con + 0.3, '0.24', fontsize = 15, fontweight = 600)
axs[1].set_ylabel('r', fontsize = 20, fontweight = 600, rotation = 0)


axs[2].plot(d, (2 * d + 0.25), c = 'darkred', label = 'LM: $Y = {1/2} \cdot r - 0.25$')
axs[2].plot(d, (-0.25 * d + 10.5)/(3), c = 'darkblue', label = 'IS: $Y = 2 + 0.75 \cdot (Y - 2) + -3 \cdot r + 10$')
axs[2].plot(p_mon_exp, j_mon_exp, 'go', c = 'black')
axs[2].set_title('Política Monetária Expansionista', fontsize = 20, fontweight = 700)
axs[2].legend(loc = 'upper center',
              bbox_to_anchor = (1.2, 0.5),
              fancybox = True,
              fontsize = 10)
axs[2].text(p_mon_exp - .1, j_mon_exp + 0.3, '1.56', fontsize = 15, fontweight = 600)
axs[2].set_xlabel('Y', fontsize = 20, fontweight = 600)
plt.tight_layout()
plt.show()