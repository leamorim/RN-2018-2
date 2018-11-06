# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:51:36 2018

@author: Lucas Amorim

Script de Treinamento da Rede, inicialmente com uma Rede Neural
"""
import pandas as pd
#import numpy as np
from sklearn.cross_validation import train_test_split

#Leitura dos dados de Treinamento
data_set = pd.read_csv('treinamento.csv')

##Aqui já estão só os dados do conjunto de Treinamento
previsores = data_set_c1.iloc[:,1:-2].values
classe = data_set_c1.iloc[:,-1].values

# Número de features do nosso data set.
input_dim = previsores.shape[1]

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



