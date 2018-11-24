# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

from sklearn import metrics

import pandas as pd

from keras import optimizers

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_log_error

import scikitplot as skplt
import matplotlib
import matplotlib.pyplot as plt

data_set = pd.read_table('data/TRN')
data_set.drop_duplicates(inplace=True)  # Remove exemplos repetidos

# Também convertemos os dados para arrays ao invés de DataFrames
X = data_set.iloc[:,1:-2].values
y = data_set.iloc[:,-2:-1].values

## Treino: 50%, Validação: 25%, Teste: 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/3, random_state=42, stratify=y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#----------------------------------------------------------------------------#

# Número de features do nosso data set.
input_dim = X_train.shape[1]

# Aqui criamos o esboço da rede.
classifier = Sequential()

classifier.add(Dense(16, activation='tanh', input_dim=input_dim))
classifier.add(Dense(32, activation='relu'))
classifier.add(Dense(32, activation='softmax'))
classifier.add(Dense(1, activation='sigmoid'))

# Por fim compilamos o modelo especificando um otimizador, a função de custo, e opcionalmente
# métricas para serem observadas durante treinamento.
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
classifier.compile(optimizer='sgd', loss='mean_squared_error')

# Para treinar a rede passamos o conjunto de treinamento e especificamos o tamanho do mini-batch,
# o número máximo de épocas, e opcionalmente callbacks. No seguinte exemplo utilizamos early
# stopping para interromper o treinamento caso a performance não melhore em um conjunto de validação.
history = classifier.fit(X_train, y_train, batch_size=64, epochs=120, 
                         callbacks=[EarlyStopping(patience=3)], validation_data=(X_val, y_val))

#----------------------------------------------------------------------------#

def extract_final_losses(history):
    """Função para extrair o melhor loss de treino e validação.
    
    Argumento(s):
    history -- Objeto retornado pela função fit do keras.
    
    Retorno:
    Dicionário contendo o melhor loss de treino e de validação baseado 
    no menor loss de validação.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    idx_min_val_loss = np.argmin(val_loss)
    return {'train_loss': train_loss[idx_min_val_loss], 'val_loss': val_loss[idx_min_val_loss]}

def plot_training_error_curves(history):
    """Função para plotar as curvas de erro do treinamento da rede neural.
    
    Argumento(s):
    history -- Objeto retornado pela função fit do keras.
    
    Retorno:
    A função gera o gráfico do treino da rede e retorna None.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    fig, ax = plt.subplots()
    ax.plot(train_loss, label='Train')
    ax.plot(val_loss, label='Validation')
    ax.set(title='Training and Validation Error Curves', xlabel='Epochs', ylabel='Loss (MSE)')
    ax.legend()
    plt.show()

def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        plt.show()
        y_pred_scores = y_pred_scores[:, 1]
        auroc = roc_auc_score(y, y_pred_scores)
        aupr = average_precision_score(y, y_pred_scores)
        performance_metrics = performance_metrics + (auroc, aupr)
    return performance_metrics

def print_metrics_summary(accuracy, recall, precision, f1, auroc=None, aupr=None):
    print()
    print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))
    print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))
    print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))
    print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))
    if auroc is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))
    if aupr is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))
        
##----------------------------------------------------------------------------#
      
plot_training_error_curves(history)

##----------------------------------------------------------------------------#

# Fazer predições no conjunto de teste
y_pred_scores = classifier.predict(X_test)
y_pred_class = classifier.predict_classes(X_test, verbose=0)
y_pred_scores_0 = 1 - y_pred_scores
y_pred_scores = np.concatenate([y_pred_scores_0, y_pred_scores], axis=1)

## Matriz de confusão
print('Matriz de confusão no conjunto de teste:')
print(confusion_matrix(y_test, y_pred_class))

## Resumo dos resultados
losses = extract_final_losses(history)
print()
print("{metric:<18}{value:.4f}".format(metric="Train Loss:", value=losses['train_loss']))
print("{metric:<18}{value:.4f}".format(metric="Validation Loss:", value=losses['val_loss']))
print('\nPerformance no conjunto de teste:')
accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, y_pred_class, y_pred_scores)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)

##----------------------------------MLP--------------------------------------# 
#
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.svm import SVC
#from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
#
#def create_sklearn_compatible_model():
#    model = Sequential()
#    model.add(Dense(20, activation='tanh', input_dim=input_dim))
#    model.add(Dense(1, activation='sigmoid'))
#    model.compile(optimizer='adam', loss='mean_squared_error')
#    return model
#
#mlp_clf = KerasClassifier(build_fn=create_sklearn_compatible_model, 
#                          batch_size=64, epochs=100,
#                          verbose=0)
#mlp_clf.fit(X_train, y_train)
#mlp_pred_class = mlp_clf.predict(X_val)
#mlp_pred_scores = mlp_clf.predict_proba(X_val)
#accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val, mlp_pred_class, mlp_pred_scores)
#print('Performance no conjunto de validação:')
#print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
#
##-------------------?Máquina de Vetores de Suporte?--------------------------#
#
#svc_clf = SVC(probability=True)  # Modifique aqui os hyperparâmetros
#svc_clf.fit(X_train, y_train)
#svc_pred_class = svc_clf.predict(X_val)
#svc_pred_scores = svc_clf.predict_proba(X_val)
#accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val, svc_pred_class, svc_pred_scores)
#print('Performance no conjunto de validação:')
#print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
#
##---------------------------Gradient Boosting--------------------------------#
#
#gb_clf = GradientBoostingClassifier()  # Modifique aqui os hyperparâmetros
#gb_clf.fit(X_train, y_train)
#gb_pred_class = gb_clf.predict(X_val)
#gb_pred_scores = gb_clf.predict_proba(X_val)
#accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val, gb_pred_class, gb_pred_scores)
#print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
#
##----------------------------Random forest-----------------------------------#
#
#rf_clf = RandomForestClassifier()  # Modifique aqui os hyperparâmetros
#rf_clf.fit(X_train, y_train)
#rf_pred_class = rf_clf.predict(X_val)
#rf_pred_scores = rf_clf.predict_proba(X_val)
#accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val, rf_pred_class, rf_pred_scores)
##print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
#
##------------------------------Ensembles-------------------------------------#
#
#mlp_ens_clf = KerasClassifier(build_fn=create_sklearn_compatible_model,
#                              batch_size=64, epochs=50, verbose=0)
#svc_ens_clf = SVC(probability=True)
#gb_ens_clf = GradientBoostingClassifier()
#rf_ens_clf = RandomForestClassifier()
#ens_clf = VotingClassifier([('mlp', mlp_ens_clf), ('svm', svc_ens_clf), ('gb', gb_ens_clf), ('rf', rf_ens_clf)], 
#                           voting='soft')
#
#ens_clf.fit(X_train, y_train)
#ens_pred_class = ens_clf.predict(X_val)
#ens_pred_scores = ens_clf.predict_proba(X_val)
#accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_val, ens_pred_class, ens_pred_scores)
#print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
#
##------------------------------Everyone-----------------------------#
#
#mlp_pred_class = mlp_clf.predict(X_test)
#mlp_pred_scores = mlp_clf.predict_proba(X_test)
#accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, mlp_pred_class, mlp_pred_scores)
#print('MLP')
#print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
#
#svc_pred_class = svc_clf.predict(X_test)
#svc_pred_scores = svc_clf.predict_proba(X_test)
#accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, svc_pred_class, svc_pred_scores)
#print('\n\nSupport Vector Machine')
#print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
#
#gb_pred_class = gb_clf.predict(X_test)
#gb_pred_scores = gb_clf.predict_proba(X_test)
#accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, gb_pred_class, gb_pred_scores)
#print('\n\nGradient Boosting')
#print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
#
#rf_pred_class = rf_clf.predict(X_test)
#rf_pred_scores = rf_clf.predict_proba(X_test)
#accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, rf_pred_class, rf_pred_scores)
#print('\n\nRandom Forest')
#print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)
#
#ens_pred_class = ens_clf.predict(X_test)
#ens_pred_scores = ens_clf.predict_proba(X_test)
#accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, ens_pred_class, ens_pred_scores)
#print('\n\nEnsemble')
#print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)