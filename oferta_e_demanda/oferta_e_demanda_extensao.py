# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:05:45 2022

@author: Ian
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

###############################################################################
# Funcoes de oferta e de demanda

# Funcao de oferta linear
def of_lin(p, o_intr = 3, o_incl = 1):
    
    return o_intr + o_incl * p

# Funcao de demanda linear
def dem_lin(p, d_intr = 6, d_incl = 2):
    
    return d_intr - d_incl * p

# Funcoes nao lineares quaisquer
def of_log(p, o_coef1):
    
    return np.log(p) * p + o_coef1

def dem_exp(p, d_coef1, d_bias):
    
    return np.exp(-p) - d_coef1 * np.log(p) + d_bias

###############################################################################
# Funcoes de equilibrio com Impostos

# Encontra os precos e quantidades de equilibrio
# para um mercado com oferta e demanda lineares
# e imposto ad valorem
def eq_ad_val_analitico_lin(d_intr, o_intr, d_incl, o_incl, imp):
    
    """
    d_intr: intercepto da demanda
    o_intr: intercepto da oferta
    d_incl: inclinacao da demanda
    o_incl: inclinacao da oferta
    imp   : % de impostos
    """
    # Calcula o preco da oferta e da demanda de equilibrio
    ps = (d_intr - o_intr) / (o_incl + d_incl * (1 + imp))
    pd = ps * (1 + imp)
    
    # Calcula as quantidades de equilibrio
    qd = d_intr - d_incl * pd
    qs = o_intr + o_incl * ps
    
    return ps, pd, qs, qd

# Calcula preco de equilibrio analiticamente para impostos por
# quantidade
def eq_quant_analitico_lin(d_intr, o_intr, d_incl, o_incl, imp):
    
    """
    d_intr: intercepto da demanda
    o_intr: intercepto da oferta
    d_incl: inclinacao da demanda
    o_incl: inclinacao da oferta
    imp   : % de impostos
    """
    
    # Calcula o preco da oferta e da demanda de equilibrio
    ps = (d_intr - d_incl * imp - o_intr) / (o_incl + d_incl)
    pd = ps_eq + imp

    # Calcula as quantidades de equilibrio
    qd = d_intr - d_incl * pd
    qs = o_intr + o_incl * ps
    
    return ps, pd, qs, qd

# Calcula os precos e quantidades de equilibrio
# dadas funcoes de oferta e demanda atraves de
# tecncias de ponto fixo
def eq_imp_grid(f_of, f_dem, imp, pe, ps = 0, pt = 1000, ad_valorem = True):
    
    """
    f_of      : funcao de oferta
    f_dem     : funcao de demanda
    imp       : imposto
    ps        : primeiro ponto do grid (default = 0)
    pe        : ultimo ponto do grid
    ad_valorem: se o imposto sera advalorem ou por quantidade (default = True)
    """
    
    if ad_valorem:
        assert imp >= 0 and imp <= 1
    
    # Cria vetor de precos e excesso de demanda
    p_vec = np.linspace(ps, pe, pt)
    val_excesso_dem = []
    
    # Para cada preco no vetor
    for ps in p_vec:
        
        # Preco pago pela demanda eh o preco
        # recebido pela oferta acrecido do imposto
        if ad_valorem:
            pd = (1 + imp) * ps
            
        else:
            pd = ps + imp
        
        Qd = f_dem(pd)
        Qs = f_of(ps)
        
        # Calcula excesso de demanda
        excesso = abs(Qd - Qs)
        val_excesso_dem.append(excesso)
    
    # Encontra equilibrio
    val_excesso_dem = np.array(val_excesso_dem)
    ps_eq = p_vec[np.argmin(val_excesso_dem)]
    
    if ad_valorem:
        pd_eq = (1 + imp) * ps_eq
    
    else:
        pd_eq = ps_eq + imp
    
    qs_eq = f_of(ps_eq)
    qd_eq = f_dem(pd_eq)
    
    #assert round(qs_eq, 2) == round(qd_eq, 2)
    
    return ps_eq, pd_eq, qs_eq, qd_eq

# Algoritimo que implementa o algoritimo da bissecao
def bissecao(f, x1, x2, tol = 4, **params):
    
    """
    f       : funcao que queremos a raiz
    x1      : valor que determina imagem positiva
    x2      : valor que determina imagem negativa
    tol     : tolerancia no arredondamento
    **params: kwargs para a funcao
    """
    
    # Busca um valor que leve a funcao
    # a uma imagem negativa
    i = 0
    while f(x2, **params) >= 0 and i < 100:
        
        x2 += 2
        i += 1
    
    # Caso na iteracao acima i == 100
    # assumimos que nao encontramos um
    # valor que leve a funcao para uma
    # imagem negativa
    assert i < 100
    
    # Media dos valores
    xm = (x1 + x2) / 2
    
    # Enquanto nao encontrarmos a raiz
    while round(abs(f(xm, **params)), tol) > 0:
        
        if f(xm, **params) > 0:
            
            x1 = xm
            
        elif f(xm, **params) < 0:
            
            x2 = xm
            
        xm = (x1 + x2) / 2
    
    return xm
    
# Funcao encontra o excesso de demanda para 
# quaisquer funcoes de oferta e demanda
def exc_de_dem(ps, **params):
    
    """
    ps : preco da oferta
    **params: parametros para as funcoes oferta ('f_of') e 
              demanda ('f_dem').
              1. Paramentros da oferta devem comecar com a
              letra 'o' e parametros da demanda devem 
              comecar com a letra 'd'.
              
              2. 'ad_valorem' eh o parametro booleano que 
              indica se estamos lidando com um imposto ad_valorem
              ou por quantidade.
              
              3. 'imp' eh o parametro que da o imposto.
    """
    
    # Preco da demanda se o imposto for por quantidade
    pd = ps + params['imp']
    
    # Preco da demanda se o imposto for por quantidade
    if params['ad_valorem']:
        
        pd = ps * (1 + params['imp'])
    
    # Dicionarios para os parametros da funcoes oferta e demanda
    params_oferta = {}
    params_demanda = {}
    
    # Para cada parametro e seu respectivo valor
    for key, value in params.items():
        
        # Se comecar com d eh da demanda
        if key[0] == 'd':
            
            params_demanda[key] = value
            
        # Se comecar com o eh da oferta
        elif key[0] == 'o':
            
            params_oferta[key] = value
        
    # Calcula demanda e oferta para os precos
    Qd = params['f_dem'](pd, **params_demanda)
    Qs = params['f_of'](ps, **params_oferta)
    
    return Qd - Qs

# Calcula o equilibrio com impostos para qualquer 
# oferta e demanda por meio do metodo da bissecao
def eq_imp(ps1, ps2, **params):
    
    # Se for ad valorem garantir que esta entre
    # 0 e 1
    if params['ad_valorem']:
        assert params['imp'] >= 0 and params['imp'] <= 1
       
    # Preco da oferta de equilibrio pela bissecao
    ps_eq = bissecao(exc_de_dem, ps1, ps2, **params)
        
    # Calcula preco da demanda de equilibrio
    if params['ad_valorem']:
        pd_eq = (1 + params['imp']) * ps_eq
    
    else:
        pd_eq = ps_eq + params['imp']
    
    params_oferta = {}
    params_demanda = {}
    
    # Encontra parametros da oferta e da demanda
    for key, value in params.items():
        
        if key[0] == 'd':
            
            params_demanda[key] = value
            
        elif key[0] == 'o':
            
            params_oferta[key] = value
        
    # Calcula as quantidades de equilibrio
    Qd_eq = params['f_dem'](pd_eq, **params_demanda)
    Qs_eq = params['f_of'](ps_eq, **params_oferta)
    
    #assert round(qs_eq, 2) == round(qd_eq, 2)
    
    return ps_eq, pd_eq, Qs_eq, Qd_eq

###############################################################################
# Aplicacoes

res_lin = eq_imp(0.5, 1.1, f_of = of_lin, f_dem = dem_lin, o_intr = 3,
                 o_incl = 1, d_intr = 6, d_incl = 2, imp = 0.2,
                 ad_valorem = True)

res_lin2 = eq_imp(0.5, 1.1, f_of = of_lin, f_dem = dem_lin, o_intr = 3,
                 o_incl = 1, d_intr = 6, d_incl = 2, imp = 0.2,
                 ad_valorem = False)

res_n_lin = eq_imp(1, 5, f_of = of_log, f_dem = dem_exp, o_coef1 = 3,
                   d_coef1 = 5, d_bias = 13, imp = 0.3, ad_valorem = False)

###############################################################################
# Graficos e Curva de Laffer

# Plot nao linear

p_vec = np.arange(1, 10, 0.1)
qs = of_log(p_vec, 3)
qd = dem_exp(p_vec, 5, 13)

plt.plot(p_vec, qs, c = 'orange', label = 'Oferta')
plt.plot(p_vec, qd, c = 'purple', label = 'Demanda')
plt.plot(res_n_lin[0], res_n_lin[2], 'go', c = 'black', label = 'Equilíbrio')
plt.plot([res_n_lin[0], res_n_lin[0]], [0,  res_n_lin[2]],
         c = 'darkslategrey', linestyle = 'dashed')
plt.plot([0, res_n_lin[0]], [res_n_lin[2],  res_n_lin[2]],
         c = 'darkslategrey', linestyle = 'dashed')
plt.xlabel('Preço', fontsize = 15)
plt.ylabel('Quantidade', fontsize = 15)
plt.title('Equilíbrio para Funções não Lineares', fontsize = 25)
plt.legend(loc = 'best')
plt.text(res_n_lin[0], res_n_lin[2] * 1.1,
         '({num1: .2f}, {num2: .2f})'.format(num1 = res_n_lin[0], num2 = res_n_lin[2]))
plt.show()
