# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:48:35 2022

@author: Ian

Equilibrio Geral Scarf
"""

import numpy as np
import matplotlib.pyplot as plt


###############################################################################
# Funcoes auxiliares

# A demanda de cada agente por cada bem aos precos dados
def demanda(pessoa, bem, valor_dot, a, b, p):
    
    """
    Calcula a demanda de uma pessoa por um bem
    ____________________________
    pessoa    : indice da pessoa cuja demanda sera calculada (0 a n)
    bem       : indice do bem cuja demanda sera calculada (0 a k)
    valor_dot : o valor da dotacao da pessoa
    a         : matriz de preferencias das pessoas
    b         : vetor da substitutibilidade entre os bens para as pessoas
    p         : vetor de precos
    """
    
    # Intensidade de preferencia pelo bem especifico
    # da pessoa especifica
    a_ij = a[pessoa][bem]
    
    # Substitutibilidade entre commodities da pessoa
    # especifica
    b_i = b[pessoa]
    
    # Elevando os precos a um expoente usado no calculo
    p_1b = p ** (1 - b_i)
    
    valor_denom = 0
    
    # Calculando o valor de uma parcela do denominador
    for i in range(a.shape[1]):
        
        valor_denom += p_1b[i] * a[pessoa][i]
        
    num = a_ij * valor_dot
    
    den = (p[bem] ** b_i) * valor_denom
    
    # Impedir que o denominador seja 0
    if abs(den) < 1e-12:
        
        print(valor_denom, den)
        den = max(den, 1e-12)
    
    return num / den

# O excesso de demanda de mercado de um bem aos precos dados
def excessoDemMercado(bem, p, w, a, b):
    
    """
    Calcula o excesso de demanda de mercado por um bem
    ____________________________
    bem : indice do bem que tera o excesso de demanda de mercado calculado
          (0 a k)
    p   : vetor de precos
    w   : matriz de dotacoes dos bens dos individuos
    a   : matriz de preferencias das pessoas
    b   : vetor da substitutibilidade entre os bens para as pessoas
    """
    
    # Dimensoes do vetor de precos e dotacoes devem 
    # ser compativeis
    assert p.shape[0] == w.shape[1]
    
    # Retorna um vetor coluna com os valores das dotacoes
    # dos agentes
    valor_dots = np.dot(w, p)
    
    excesso_demandas = []
    
    # Para cada pessoa
    for pessoa in range(a.shape[0]):
        
        # Calcula sua demanda
        dem = demanda(pessoa, bem, valor_dots[pessoa], a, b, p)
        
        # Calcula seu excesso de demanda
        ex_dem = dem - w[pessoa][bem]
        
        excesso_demandas.append(ex_dem)
    
    # O excesso de demanda de mercado sera a soma
    # dos excessos de demanda de cada agente
    excesso_dem_mercado = sum(excesso_demandas)
    
    return excesso_dem_mercado

# Cria um vetor com os excessos de demanda de mercado por
# cada bem na economia
def excessoDemVec(n_bens, p, w, a, b):
    
    """
    Forma um vetor com os excessos de demanda de mercado para cada mercado
    ____________________________
    n_bens : numero de mercados na economia
    p      : vetor de precos
    w      : matriz de dotacoes dos bens dos individuos
    a      : matriz de preferencias das pessoas
    b      : vetor da substitutibilidade entre os bens para as pessoas
    """
    
    # Cria vetor
    excesso_dem_vec = []
    
    # Itera por cada bem na economia, gerando o
    # excesso de demanda de mercado
    for i in range(n_bens):
        
        ex_dem = excessoDemMercado(i, p, w, a, b)
        excesso_dem_vec.append(ex_dem)
        
    return excesso_dem_vec

# Verificar se tem que usar o k (numero pequeno)
# ou nao
def brower(p, ex_dem_vec):
    
    normaliza = 1
    novo_p = p.copy()
    
    for i in range(len(ex_dem_vec)):
        
        normaliza += max(0, ex_dem_vec[i])
        novo_p[i] += max(0, ex_dem_vec[i])
        
    
    return novo_p / normaliza

###############################################################################
# Funcoes Scarf

# Formacao da matriz inicial
def matrizInicial(D, n_bens):
    
    """
    Funcao forma a matriz inicial que sera utilizada para formar o 
    primeiro conjunto primitivo. Essa matriz contem n vetores coluna 
    correspondentes aos n lados do simplex e 1 vetor coluna interior
    ao simplex.
    ____________________________
    D      : valor total a que cada vetor interior ao simplex deve somar
    n_bens : numero de mercados na economia
    """
    
    # Estabelece o valor arbitrariamente grande
    M = D + n_bens + 10
    
    # Estabelece a matriz dos lados do simplex
    matriz = [[M - j if j != i else 0 for j in range(n_bens)] 
              for i in range(n_bens)]
    
    # Primeiro vetor interno ao simplex
    vec = [[D - (n_bens - 1) if i == 0 else 1] for i in range(n_bens)]
    
    # Concatenando a matriz com o vetor interno para formar a matriz inicial
    matriz = np.c_[matriz, vec]
    
    matriz = matriz.astype(float)
    
    return matriz

def labels2(p, ex_dem_vec):
    
    """
    Uma forma de determinar as labels: menor indice tal que f(x) - x 
    seja positivo
    ____________________________
    p          : vetor de precos
    ex_dem_vec : excessos de demanda de mercado a esses precos
    """
    
    # Faz a diferenca entre o novo mapeamento de 
    # precos e o vetor de precos original
    vec_labels = brower(p, ex_dem_vec) - p
    
    label = 1
    
    for i in range(len(vec_labels)):
        
        # Se componente for positivo
        if vec_labels[i] >= 0:
            
            label += i
            break
    
    return label

def labels(p, ex_dem_vec):
    
    """
    Outra forma de determinar as labels: sera o argumento que maximiza o 
    quociente entre o vetor de excessos de demanda e os precos
    ____________________________
    p          : vetor de precos
    ex_dem_vec : excessos de demanda de mercado a esses precos
    """
    
    label = 1 + np.argmax(ex_dem_vec / p)
    
    return label

# Determina a lista inicial de labels
def listaLabels(func_l, n_bens, mat_in, ex_dem_vec):
    
    """
    Determina a lista inicial de labels
    ____________________________
    func_l     : funcao usada para atribuir labels
    n_bens     : numero de mercados na economia
    mat_in     : matriz inicial (criada na funcao matrizInicial)
    ex_dem_vec : excessos de demanda de mercado a esses precos
    """
    
    # As labels dos lados sao i = 1, ..., n
    lista_labels = [i + 1 for i in range(n_bens)]
    
    # Determina a label do primeiro vetor interno
    label = func_l(mat_in[:, -1], ex_dem_vec)
    
    lista_labels.append(label)
    
    return lista_labels

def substituicao(prim, l_prim, D, mat_in):
    
    """
    Realiza a operacao de substituicao, permitindo encontrar um novo 
    conjunto primitivo a partir da remocao de um vetor do conjunto 
    primitivo anterior e do deslocamento do subsimplex de modo paralelo
    ao lado que continha o vetor removido. A presente funcao encontra
    o vetor a ser removido do subsimplex atual e propoe o novo vetor
    de modo a formar um novo subsimplex.
    ____________________________
    prim   : conjunto primitivo atual
    l_prim : labels do primitivo atual (ultimo label da lista OBRIGATORIAMENTE
             sera o label do vetor que entrou por ultimo no primitivo)
    D      : valor total a que cada vetor interior ao simplex deve somar
    mat_in : matriz inicial (criada na funcao matriz Inicial). Como sera
             visto na operacao da funcao scarf, todo novo vetor que sera
             proposto pela funcao substituicao sera concatenada a matriz
             inicial, de modo que o input desse parametro se altera a cada
             novo vetor proposto.
    """
    
    # Criando uma matriz com as mesmas dimensoes do
    # conjunto primitivo que tenha 1 nos menores elementos 
    # de cada linha, 2 no segundo menor elemento e 0 caso contrario
    mat_menores = np.zeros(shape = prim.shape)
    
    for i in range(mat_menores.shape[0]):
        
        # Menor valor da linha
        min_lin = min(prim[i, :])
        
        # Lembrando a regra de quebra de empate:
        # se dois vetores tiverem o mesmo indice
        # os vetores que foram adicionados a menos
        # tempo terao componentes que serao tratados
        # como menores que os dos que vieram antes,
        # mesmo se o valor for igual. Por isso pegamos
        # o ultimo indice onde aparece o menor valor.
        
        # Imagine que o n-esimo componente da coluna i
        # eh um menor da linha. Entao, a aresta n do subsimplex
        # passara pela coluna i.
        aresta = np.where(prim[i, :] == min_lin)[0][-1]
        
        # Segundo menor elemento da linha
        seg_min_lin = sorted(prim[i, :])[1]
        
        # Realizando a operacao de quebra de empate
        # Suponhamos que o menor e o segundo menor elemento
        # da linha tem o mesmo valor
        # pos_n_aresta = possivel nova aresta
        if min_lin == seg_min_lin:
            
            # O ultimo vetor a entrar sera considerado menor e recebera
            # 1 na mat_menores (mesmo que os componentes tenham o mesmo
            # valor). O indice do segundo menor valor da linha, ou seja,
            # do vetor que pode possivelmente tomar a aresta do menor da
            # linha, sera o indice do penultimo elemento a entrar no
            # primitivo com o valor min_lin = seg_min_lin
            pos_n_aresta = np.where(prim[i, :] == seg_min_lin)[0][-2]
            
        else:
            
            # Agora, caso os valores do menor da linha sejam diferentes
            # do valor do segundo maior da linha, entao o indice do vetor
            # que podera assumir a aresta sera simplesmente o ultimo vetor
            # a entrar no primitivo que tenha o valor de seg_min_lin
            pos_n_aresta = np.where(prim[i, :] == seg_min_lin)[0][-1]
            
        # Menor da linha recebe 1
        mat_menores[i, aresta] = 1
        
        # Segundo menor da linha recebe 2
        mat_menores[i, pos_n_aresta] = 2
    
    # Label do vetor que acabou de entrar no primitivo
    label_novo = l_prim[-1]
    
    # Indice do vetor a ser substituido
    ind_vec_sub = l_prim[:-1].index(label_novo)
    
    # Linha em que o vetor que ira ser substituido
    # tem o menor valor da linha
    min_lin_sub = np.where(mat_menores[:, ind_vec_sub] == 1)[0][0]
    
    # Encontrando o vetor no conjunto primitivo
    # que ira assumir a aresta do vetor a ser
    # substituido no conjunto primitivo
    vec_sub = np.where(mat_menores[min_lin_sub, :] == 2)[0][0]
    
    # Encontrando o componente do vetor que ira assumir a aresta do
    # vetor a ser substituido que eh o menor da sua linha
    min_comp_sub = np.where(mat_menores[:, vec_sub] == 1)[0][0]
    
    # Vetor que contera as restricoes para o novo vetor a ser inserido
    # no conjunto primitivo. Para cada elemento xi em vec_cond e cada
    # elemento ai no novo vetor a ser inserido no conjunto primitivo
    # ai > xi. 
    vec_cond = []
    
    # Para cada linha
    for i in range(mat_menores.shape[0]):
        
        # Se a linha for a linha do vetor a ser
        # substituido adicionar as condicoes o
        # novo menor elemento da linha (pos_n_aresta)
        if i == min_lin_sub:
            
            vec_cond.append(prim[i, vec_sub])
        
        # Adiciona valor arbitrario ao componente do novo
        # vetor que devera ser maximizado
        elif i == min_comp_sub:
            
            vec_cond.append(D)
        
        else:
            
            vec_cond.append(prim[i, np.where(mat_menores[i, :] == 1)[0][0]])
    
    vec_cond = np.array(vec_cond)
    
    # Gerando o novo vetor a ser introduzido a matriz
    novo_vec = vec_cond + 1
    novo_vec[min_comp_sub] = D - novo_vec[novo_vec != D + 1].sum()
    
    # Caso houver um componente 0 em uma nova coluna a ser adicionada ao
    # subsimplex, sera como se tivessemos voltado para uma das arestas do
    # simplex. Assim, a nova coluna proposta sera justamente aquela aresta.
    
    zero_in = np.where(novo_vec == 0)[0]
    
    if len(zero_in) != 0:
        
        novo_vec = mat_in[:, zero_in[0]]
        
    return novo_vec, ind_vec_sub

def scarf(func_l, w, a, b, D):
    
    """
    A presente funcao realiza o algoritmo de Scarf ate que seja
    atingida a condicao de parada, tendo como resultado um conjunto
    primitivo que tenha em seu interior o ponto fixo. A matriz inicial
    e lista de labels sao gerados. O primeiro conjunto primitivo e suas
    labels sao formadas e a operacao de substituicao eh realizada ate que
    a condicao de parada seja atingida.
    ____________________________
    func_l : funcao usada para atribuir labels
    w      : matriz de dotacoes dos bens dos individuos
    a      : matriz de preferencias das pessoas
    b      : vetor da substitutibilidade entre os bens para as pessoas
    D      : valor total a que cada vetor interior ao simplex deve somar
    """
    
    # Ve o numero de commodities
    n_bens = w.shape[1]
    
    # Gera a matriz inicial
    mat_in = matrizInicial(D, n_bens)
    
    # Gera o primeiro vetor de excesso de demandas
    # O vetor de precos inicial sera o primeiro ponto
    # interno ao simplex
    ex_dem_vec = excessoDemVec(n_bens, (mat_in[:, -1] / D), w, a, b)
    
    # Gera a lista inicial de labels
    labels = listaLabels(func_l, n_bens, mat_in, ex_dem_vec)
    
    # Conjunto primitivo inicial
    prim = mat_in[:, 1:]
    
    # Labels primitivo inicial
    l_prim = labels[1:]
    
    # Lista dos primitivos
    lista_prim = [prim]
    
    # Lista dos excessos de demanda
    ex_dem_lista = [ex_dem_vec]
    
    i = 1
    
    # Enquanto nao for alcancado um primitivo
    # que tenha vetores com todas as labels
    # de 1 a n_bens (Condicao de Parada)
    while len(np.unique(l_prim)) != n_bens:
        
        i+= 1
        
        # Encontra o novo vetor e o indice a remover
        novo_vec, ind_remover = substituicao(prim, l_prim, D, mat_in)
        
        # IMPORTANTE: Na iteracao do algoritmo os novos
        # vetores sao adicionados sempre ao final tanto
        # da lista de vetores (mat_in) e, mais importante,
        # sempre ao final do conjunto primitivo. A label do
        # novo vetor tambem eh adicionada ao final da lista
        # de labels e da lista de labels dos primitivos. 
        # O funcionamento do algoritmo depende disso. 
        
        # Adiciona o vetor novo na lista de vetores
        mat_in = np.c_[mat_in, novo_vec]
        
        # Remove do primitivo o vetor com label repetida
        prim = np.delete(prim, ind_remover, 1)
        
        # Adiciona o novo vetor ao primitivo
        prim = np.c_[prim, novo_vec]
        
        lista_prim.append(prim)
        
        # Excesso de demanda do novo vetor de precos
        ex_dem_vec = excessoDemVec(n_bens, novo_vec, w, a, b)
        ex_dem_lista.append(ex_dem_vec)
        
        # Encontra o label do novo vetor
        novo_l = func_l(novo_vec, ex_dem_vec)
        
        # Apensa o label a lista de labels
        labels.append(novo_l)
        
        # Remove o label do vetor removido
        l_prim.pop(ind_remover)
        
        # Apensa o novo label ao final da lista de labels do
        # primitivo - necessario para o funcionamento
        l_prim.append(novo_l)
    
    return prim, l_prim, ex_dem_lista, lista_prim, labels, mat_in, i

###############################################################################
# Funcoes Metodo de Newton

def aproxDerivadaExcessoDemMercado(bem, p, w, a, b, var, tol = 4):
    
    # Iterador
    i = 1
    
    h = 0.1 ** i
    
    # Guardando o vetor x inicial
    p_vec_i = p.copy()
    
    # Criando o vetor x novo
    p_vec_n = p.copy()
    p_vec_n[var] = p_vec_n[var] + h
    
    # Aproximacao da derivada em x para o
    # valor inicial de h
    dif = (excessoDemMercado(bem, p_vec_n, w, a, b) -
           excessoDemMercado(bem, p_vec_i, w, a, b)) / h
    
    # Variavel que armazena os valores anteriores
    # de dif
    temp = 0
    
    while round(abs(dif - temp), tol) != 0:
        
        # Reduzo h
        i += 1
        h = 0.1 ** i
        
        # Atualizo dif
        temp = dif
        
        # Atualizo o valor da variavel
        p_vec_n = p.copy()
        p_vec_n[var] = p_vec_n[var] + h
        
        # Aproximacao da derivada em x para o
        # valor inicial de h
        dif = (excessoDemMercado(bem, p_vec_n, w, a, b) -
               excessoDemMercado(bem, p_vec_i, w, a, b)) / h
        
    
    return dif

def jacobianoExcessoDemMercado(n_bens, p, w, a, b, tol = 4):
    
    jacobiano = []
    
    # Para cada variavel
    for var in range(n_bens):
        
        linha = []
        
        # Para cada funcao excesso de demanda de mercado
        for f in range(n_bens):
            
            # Calcula o valor derivada da funcao excesso de demanda de
            # mercado para o bem f com respeito a variavel var para
            # os valores p.
            dfi_dxi = aproxDerivadaExcessoDemMercado(f, p, w, a, b, var, tol)
            
            # Forma a linha da jacobiana
            linha.append(dfi_dxi)
        
        # apensa a linha na matriz
        jacobiano.append(linha)

    return np.array(jacobiano)

def newton(n_bens, p, w, a, b, tol = 4):
    
    # Cria a matriz jacobiana
    jacobiana = jacobianoExcessoDemMercado(n_bens, p, w, a, b)
    
    # Copia o vetor de precos inicial
    p_vec_i = p.copy()
    
    ex_dem_list = []
    
    # Encontra o vetor de excessos de demanda a esses precos
    ex_dem_vec = excessoDemVec(n_bens, p, w, a, b)
    ex_dem_list.append(ex_dem_vec)
    
    # Inverte a jacobiana
    assert np.linalg.det(jacobiana) != 0
    inv_jacobiana = np.linalg.inv(jacobiana)
    
    # Primeira iteracao
    p_vec_n = p_vec_i - np.dot(inv_jacobiana, ex_dem_vec)
    
    # Calcula a distancia euclidiana
    dist = np.linalg.norm(p_vec_n - p_vec_i)
    
    # Enquanto a distancia euclidiana entre os vetores em duas
    # iteracoes subsequentes nao for aproximadamente 0
    while round(abs(dist), tol) != 0:
        
        # Substitui qual sera o vetor da iteracao anterior
        p_vec_i = p_vec_n
        
        # Calcula a nova matriz jacobiana
        jacobiana = jacobianoExcessoDemMercado(n_bens, p_vec_n, w, a, b)
        
        # Encontra o vetor de excessos de demanda a esses precos
        ex_dem_vec = np.array(excessoDemVec(n_bens, p_vec_n, w, a, b))
        ex_dem_list.append(ex_dem_vec)
        
        # Inverte a jacobiana
        assert np.linalg.det(jacobiana) != 0
        inv_jacobiana = np.linalg.inv(jacobiana)
        
        # Itera
        p_vec_n = p_vec_i - np.dot(inv_jacobiana, ex_dem_vec)
        
        # Calcula a distancia euclidiana
        dist = np.linalg.norm(p_vec_n - p_vec_i)
    
    return p_vec_n, ex_dem_list
     
###############################################################################
# Implementa funcao principal

def main(func_l, w, a, b, D, tol):
        
    # Aproxima o ponto fixo por Scarf
    resultado_scarf = scarf(func_l, w, a, b, D)
    ex_dem_scarf = resultado_scarf[2]
    
    # Encontra o vetor de precos aproximado pelo metodo de
    # Scarf pela media entre todos os pontos no primitivo
    p = np.mean(resultado_scarf[0], 1)
    
    n_bens = len(p)
    
    # Aplica o metodo de Newton para refinar a aproximacao
    ponto_fixo, ex_dem_newton = newton(len(p), p, w, a, b, tol)
    ponto_fixo_div = ponto_fixo / D
    
    ex_dem_scarf.extend(ex_dem_newton)
    
    # Cria uma lista que mostra os excessos de demanda
    # por bem
    ex_dem_por_bem = []
    
    # Para cada bem
    for bem in range(n_bens):
        
        lista_ex_dem_por_bem = []
        
        # Para cada lista de excesso de demanda
        for ex_dem_vec in ex_dem_scarf:
            
            lista_ex_dem_por_bem.append(ex_dem_vec[bem])
        
        ex_dem_por_bem.append(lista_ex_dem_por_bem)
    
    # Creating plot
    fig = plt.figure()
    ax = plt.subplot(111)
    
    for i in range(len(ex_dem_por_bem)):
        
        ax.plot(ex_dem_por_bem[i],
                 label = 'Bem {f}'.format(f = i + 1))
    
    ax.plot(np.arange(0, len(ex_dem_por_bem[0])),
            np.repeat(0, len(ex_dem_por_bem[0])),
            color = 'black', label = 'Zero', linewidth = 1)
    ax.set_title('Convergência dos Excessos de Demanda')
    ax.set_xlabel('Iterações')
    ax.set_ylabel('Excesso de Demanda')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    return ponto_fixo_div, ex_dem_scarf, ex_dem_por_bem

###############################################################################
# Exemplos

# Dotacoes
w = [[0.6, 0.2, 0.2, 20, 0.1, 2, 9, 5, 5, 15],
     [0.2, 11, 12, 13, 14, 15, 16, 5, 5, 9],
     [0.4, 9, 8, 7, 6, 5, 4, 5, 7, 12],
     [1, 5, 5, 5, 5, 5, 5, 8, 3, 17],
     [8, 1, 22, 10, 0.3, 0.9, 5.1, 0.1, 6.2, 11]]

w = np.array(w)

w2 = np.array([[0.6, 0.2, 0.2],
               [0.2, 11, 12],
               [0.4, 9, 8],
               [1, 5, 5],
               [8, 1, 22]])

# Intensidade de preferencia por uma commoditie
a = [[1, 1, 3, 0.1, 0.1, 1.2, 2, 1, 1, 0.7],
     [1, 1, 1, 1, 1, 1,1, 1, 1, 1],
     [9.9, 0.1, 5, 0.2, 6, 0.2, 8, 1, 1, 0.2],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     [1, 13, 11, 9, 4, 0.9, 8, 1, 2, 10]]

a = np.array(a)

a2 = np.array([[1, 1, 3],
               [1, 1, 1],
               [9.9, 0.1, 5],
               [1, 2, 3],
               [1, 13, 11]])

# Substitutibilidade das commodities
b = [2, 1.3, 3, 0.2, 0.6]

#aaaa = scarf(labels, w2, a2, b, 1000)

teste2 = scarf(labels, w, a, b, 250)

res = main(labels, w, a, b, 250, 4)
