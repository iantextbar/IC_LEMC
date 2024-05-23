# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 16:51:44 2022

@author: Ian

Matching
"""
###############################################################################

# Funcoes ----

def organiza_propostas(lista_preferencias, n_pessoas_g2):
    
    """
    Com base na lista de preferencias do primeiro grupo, na lista
    de membros do primeiro grupo e no numero de pessoas no segundo 
    grupo, forma uma lista com as propostas que cada pessoa do
    segundo grupo recebeu. O primeiro elemento da lista representa
    as propostas que a primeira pessoa do segundo grupo recebeu e
    assim por diante. 
    _________________________________
    
    lista_preferencias: lista de preferencias do primeiro grupo
    n_pessoas_g2: o numero de pessoas no segundo grupo
    
    """
    
    organiza_propostas = []
    solteiro = []
    
    # Preenche organiza propostas com listas vazias na 
    # mesma quantidade que existem pessoas que recebem propostas
    for preferencia in range(n_pessoas_g2):
        
        organiza_propostas.append([])
    
    # Para cada pessoa que lanca propostas, recebe sua pessoa mais preferida
    # e apensa a lista de propostas recebidas de sua pessoa mais preferida
    # o numero correspondente a ela
    for pessoa in range(len(lista_preferencias)):
        
        mais_preferido = lista_preferencias[pessoa][0]
        
        if mais_preferido == 0:
            
            solteiro.append(pessoa + 1)
        
        else:
            organiza_propostas[mais_preferido - 1].append(pessoa + 1)
    
    return organiza_propostas, solteiro


def casamentos_e_rejeicoes(organiza_prop, solteiros, lista_pref2, n_pessoas_g1):
    
    """
    Recebe o vetor de propostas e o vetor de preferencias do
    segundo grupo e retorna os pareamentos preliminares entre
    os grupos e a lista de quais pessoas do primeiro grupo
    foram rejeitadas
    _________________________________
    
    organiza_prop: lista das propostas criada pela funcao organiza_propostas
    solteiros: lista de pessoas solteiras
    lista_pref2: a lista de preferencias dos membros do segundo grupo
    n_pessoas_g1: numero de pessoas no grupo 1
    
    """
    
    casamento_prel = []
    nova_lista_rej = [False for i in range(n_pessoas_g1)]
    
    # Para cada lista de propostas
    for proposta in range(len(organiza_prop)):
        
        # Busca na lista de preferencias da pessoa que
        # recebeu as propostas, partindo das pessoas
        # mais preferidas para as menos preferidas
        for preferencia in lista_pref2[proposta]:
                        
            # Faz o casamento preliminar entre a melhor 
            # proposta e a pessoa que recebeu as propostas
            if preferencia in organiza_prop[proposta]:
                
                casamento_prel.append([preferencia, proposta + 1])
                
                # As demais opcoes que nao a melhor proposta
                # sao rejeitadas
                for pessoa in organiza_prop[proposta]:
                    
                    if pessoa != preferencia:
                        
                        nova_lista_rej[pessoa - 1] = True
                
                break
            
            # Se a pessoa prefere ficar solteira
            elif preferencia == 0:
                
                casamento_prel.append([preferencia, proposta + 1])
                
                for pessoa in organiza_prop[proposta]:
                    
                    nova_lista_rej[pessoa - 1] = True
                
                break

    # Append das pessoas solteiras na lista de casamentos preliminares
    if len(solteiros) != 0:
            
        for solteiro in solteiros:
                
            casamento_prel.append([solteiro, 0])

    return casamento_prel, nova_lista_rej

        
def atualiza_preferencias(lista_preferencias, lista_rej):
    
    """
    Com base em quem foi rejeitado na ultima iteracao
    atualiza as preferencias das pessoas do primeiro grupo
    _________________________________
    
    lista_preferencias: a lista de preferencias a ser atualizada
    lista_rej: a lista de quem foi rejeitado na ultima iteracao
    
    """
    
    # Para cada pessoa da lista de rejeitados
    for i in range(len(lista_rej)):
        
        # Se a pessoa tiver sido rejeitada remove
        # o primeiro elemento da sua lista de preferencias
        if lista_rej[i]:
            
            lista_preferencias[i].pop(0)

    return lista_preferencias


def nomeia_casamentos(casamentos, nomes_grupo1, nomes_grupo2):
    
    """
    Transforma a lista de casamentos identificados por identificadores
    numericos em uma lista de casamentos identificados pelos nomes dos
    membros de cada grupo
    _________________________________
    
    casamentos: lista de casamentos identificados pelo identificador numerico
    nomes_grupo1: lista de strings com os nomes dos membros do grupo1
    nomes_grupo2: lista de strings com os nomes dos membros do grupo2
    
    """
    
    casamentos_nomeados = []
    
    for par in casamentos:
        
        casamentos_nomeados.append([nomes_grupo1[par[0] - 1],
                                    nomes_grupo2[par[1] - 1]])
        
    return casamentos_nomeados


def iteracao(vec_grupo1, vec_grupo2, n_igual = True):
    
    """
    Recebe os vetores de preferencias de cada um dos grupos e 
    retorna a configuracao final dos pareamentos entre os 
    grupos respeitando as restricoes
    _________________________________
    
    vec_grupo1: lista de preferencias do primeiro grupo
    vec_grupo2: lista de preferencias do segundo grupo
    n_igual: booleano se cada grupo tem o mesmo numero de membros
    
    """
    
    # Armazena a quantidade de pessoas no segundo grupo
    if n_igual:
        n_pes2 = len(vec_grupo1[0])
        n_pes1 = n_pes2
    else:
        n_pes2 = len(vec_grupo1[0]) - 1
        n_pes1 = len(vec_grupo2[0]) - 1
    
    # Lista criada para armazenar os casamentos preliminares
    # da iteracao anterior
    casa_prel_anterior = []
    
    # Cria listas de preferencias, de pessoas em cada grupo e de 
    # rejeicoes no primeiro grupo
    lp1, lp2, = vec_grupo1, vec_grupo2
    rej = [False for i in range(n_pes1)]
    
    # Organiza as primeiras propostas de casamento
    org_prop, sol = organiza_propostas(lp1, n_pes2)
    
    # Organiza os primeiros casamentos preliminares e rejeicoes
    casa_prel, rej = casamentos_e_rejeicoes(org_prop, sol, lp2, n_pes1)
    
    # Atualiza preferencias com base nas primeiras rejeicoes
    lp1 = atualiza_preferencias(lp1, rej)
    
    # Enquanto os casamentos preliminares se alterarem
    # de uma iteracao para outra, continua formando
    # novos pares
    while casa_prel != casa_prel_anterior:
        
        casa_prel_anterior = casa_prel
        org_prop, sol = organiza_propostas(lp1, n_pes2)
        casa_prel, rej = casamentos_e_rejeicoes(org_prop, sol, lp2, n_pes1)
        lp1 = atualiza_preferencias(lp1, rej)
    
    #nome_casamento = nomeia_casamentos(casamentos, nomes_grupo1, nomes_grupo2)
    
    return casa_prel

###############################################################################

# Testes ----

# 1

grupo1 = ['alfa', 'beta', 'gamma', 'delta']

alfa  = [1, 2, 3, 4]
beta  = [1, 2, 3, 4]
gamma = [2, 3, 1, 4]
delta = [3, 1, 2, 4]

vec_grupo1 = [alfa, beta, gamma, delta]

grupo2 = ['A', 'B', 'C', 'D']

A = [3, 4, 1, 2]
B = [4, 1, 2, 3]
C = [1, 2, 3, 4]
D = [4, 3, 2, 1]

vec_grupo2 = [A, B, C, D]

# 2

g1 = ['alfa', 'beta', 'gamma', 'delta', 'solteiro']

alf = [1, 2, 3, 0]
bet = [2, 1, 3, 0]
gam = [1, 0, 3, 2]
dlt = [3, 2, 0, 1]

v_g1 = [alf, bet, gam, dlt]

g2 = ['A', 'B', 'C', 'solteiro']

A2 = [2, 3, 0, 1, 4]
B2 = [1, 0, 3, 2, 4]
C2 = [4, 1, 0, 3, 2]

v_g2 = [A2, B2, C2]

# Aplica Funcoes ----

pares = iteracao(vec_grupo1, vec_grupo2)
pares_nomeados = nomeia_casamentos(pares, grupo1, grupo2)

pares2 = iteracao(v_g1, v_g2, False)
pares_nomeados2 = nomeia_casamentos(pares2, g1, g2)
