# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:51:36 2018

@author: Lucas Amorim

Script de Treinamento da Rede, inicialmente com uma Rede Neural
"""
import pandas as pd
from sklearn.cross_validation import train_test_split

#Leitura dos dados de Treinamento
treinamento = pd.read_csv('treinamento.csv')
teste = pd.read_csv('teste.csv')
validacao = pd.read_csv('validacao.csv')

'''
X_train --> Atributos de treinamento 
Y_train --> Classes referentes aos atributos de treinamento

X_validacao --> Atributos de treinamento
Y_validacao --> Classes referentes aos atributos de validação

X_teste --> Atributos de teste
Y_teste --> Classes referentes aos atributos de teste
'''

#Aqui já estão só os dados do conjunto de Treinamento
X_train = treinamento.iloc[:,1:-2].values
Y_train = treinamento.iloc[:,-1].values
#Aqui já estão só os dados do conjunto de Validação
X_validacao = validacao.iloc[:,1:-2].values
Y_validacao = validacao.iloc[:,-1].values
#Aqui já estão só os dados do conjunto de Teste
X_teste = teste.iloc[:,1:-2].values
Y_teste = teste.iloc[:,-1].values


from keras.models import Sequential
# Número de features do nosso data set.
input_dim = X_train.shape[1]

# Aqui criamos o esboço da rede.
classifier = Sequential()

# Agora adicionamos a primeira camada escondida contendo 16 neurônios e função de ativação
# tangente hiperbólica. Por ser a primeira camada adicionada à rede, precisamos especificar
# a dimensão de entrada (número de features do data set).
classifier.add(Dense(16, activation='tanh', input_dim=input_dim))

# Em seguida adicionamos a camada de saída. Como nosso problema é binário só precisamos de
# 1 neurônio com função de ativação sigmoidal. A partir da segunda camada adicionada keras já
# consegue inferir o número de neurônios de entrada (16) e nós não precisamos mais especificar.
classifier.add(Dense(1, activation='sigmoid'))

# Por fim compilamos o modelo especificando um otimizador, a função de custo, e opcionalmente
# métricas para serem observadas durante treinamento.
classifier.compile(optimizer='adam', loss='mean_squared_error')


# Para treinar a rede passamos o conjunto de treinamento e especificamos o tamanho do mini-batch,
# o número máximo de épocas, e opcionalmente callbacks. No seguinte exemplo utilizamos early
# stopping para interromper o treinamento caso a performance não melhore em um conjunto de validação.
history = classifier.fit(X_train, y_train, batch_size=64, epochs=100000, callbacks=[EarlyStopping(patience=3)], validation_data=(X_val, y_val))



