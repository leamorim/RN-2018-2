# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 00:35:32 2018

@author: Manoel Amorim
"""
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


#Arquivo teste PAKDD_GERMANO para usar depois no TKN, arquivo final do projeto
#Substituir PAKDD.txt por TKN
data_set = pd.read_table('TRN')

#Separar entre as classes 1 e 2:
data_set_c1 = data_set.loc[data_set.IND_BOM_1_1 == 1]
data_set_c2 = data_set.loc[data_set.IND_BOM_1_2 == 1] 

#Separar entre Previsores(ou os atributos) e Classe
#Classe 1

previsores_c1 = data_set_c1.iloc[:,1:-2]
classe_c1 = data_set_c1.iloc[:,-2]
#Classe 2
previsores_c2 = data_set_c2.iloc[:,1:-2]
classe_c2 = data_set_c2.iloc[:,-1]

#Separação entre Treinamento (50%), Validação(25%) e teste(25%)
#Conjuntos de Treinamento, Validação e Teste da Classe 1
p_c1_train, p_c1_test, c1_train, c1_test = train_test_split(previsores_c1, classe_c1, test_size=1/4, random_state = 42)
p_c1_train, p_c1_val, c1_train, c1_val = train_test_split(p_c1_train, c1_train, test_size=1/3, random_state = 42)
#Conjuntos de Treinamento, Validação e Teste da Classe 2
p_c2_train, p_c2_test, c2_train, c2_test = train_test_split(previsores_c2, classe_c2, test_size=1/4, random_state = 42)
p_c2_train, p_c2_val, c2_train, c2_val = train_test_split(p_c2_train, c2_train, test_size=1/3, random_state = 42)

#Sampling da Classe 2 (aumentar a quantidade de exemplos da classe 2 para o mesmo tamanho da classe 1)
#Lembrar que o valor da classe vai ser 1, é só aumentar o tamanho de c2_train e colocar 1 tb
#Sampling no Conjunto de Treinamento de C2
p_c2_train = p_c2_train.sample(n = c1_train.count(),random_state = 1,replace = True)
c2_train = c2_train.sample(n=c1_train.count(),random_state = 1, replace = True)

#Sampling no Conjunto de Validação de C2
p_c2_val = p_c2_val.sample(n = c1_val.count(),random_state = 1,replace = True)
c2_val = c2_val.sample(n=c1_val.count(),random_state = 1, replace = True)

'''
Obtendo o conjunto de Treinamento final com as classes juntas
'''
#Classe 1
zeros = pd.DataFrame(np.zeros((c1_train.count(),1)),columns = ['IND_BOM_1_2'])
c1_train = pd.concat((c1_train.reset_index(drop = True),zeros),axis=1) #concatena a coluna a direita
#Classe 2
zeros = pd.DataFrame(np.zeros((c2_train.count(),1)),columns = ['IND_BOM_1_1'])
c2_train = pd.concat((zeros,c2_train.reset_index(drop = True)),axis=1) #concatena a coluna a direita

#junta o vetor das duas classes
c_train = c2_train.append([c1_train]).reset_index(drop=True)

#Previsores Classe 1 + Previsores Classe 2
p_train = p_c1_train.append(p_c2_train).reset_index(drop=True)
#Concatenando Previsores + Classes
train = pd.concat([p_train,c_train], axis = 1).reset_index(drop=True)

#Embaralha os registros (ou as linhas)
train = train.sample(frac=1).reset_index(drop=True)
train.to_csv('treinamento.csv') #Escrevendo num CSV os dados que serão usados para treinamento


'''
Validação
'''
#Classe 1
zeros = pd.DataFrame(np.zeros((c1_val.count(),1)),columns = ['IND_BOM_1_2'])
c1_val = pd.concat([c1_val.reset_index(drop = True),zeros],axis=1) #concatena a coluna a direita
#Classe 2
zeros = pd.DataFrame(np.zeros((c2_val.count(),1)),columns = ['IND_BOM_1_1'])
c2_val = pd.concat([zeros,c2_val.reset_index(drop = True)],axis=1) #concatena a coluna a direita

#junta o vetor das duas classes
c_val = c1_val.append([c2_val]).reset_index(drop=True)

#Previsores Classe 1 + Previsores Classe 2
p_val = p_c1_val.append(p_c2_val).reset_index(drop=True)
#Concatenando Previsores + Classes
val = pd.concat([p_val,c_val], axis = 1).reset_index(drop=True)

#Embaralha as Linhas
val = val.sample(frac=1).reset_index(drop=True)
val.to_csv('validacao.csv')

'''
Teste
'''
zeros = pd.DataFrame(np.zeros((c1_test.count(),1)),columns = ['IND_BOM_1_2'])
c1_test = pd.concat([zeros,c1_test.reset_index(drop = True)],axis=1) #concatena a coluna a direita
#Classe 2
zeros = pd.DataFrame(np.zeros((c2_test.count(),1)),columns = ['IND_BOM_1_1'])
c2_test = pd.concat([c2_test.reset_index(drop = True),zeros],axis=1) #concatena a coluna a direita

#junta o vetor das duas classes
c_test = c1_test.append([c2_test]).reset_index(drop=True)

#Previsores Classe 1 + Previsores Classe 2
p_test = p_c1_test.append(p_c2_test).reset_index(drop=True)
#Concatenando Previsores + Classes
test = pd.concat([p_test,c_test], axis = 1).reset_index(drop=True)

#Embaralha as Linhas
test = test.sample(frac=1).reset_index(drop=True)
test.to_csv('teste.csv')




