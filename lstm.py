from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

base = pd.read_csv('EURUSD2019M1_treinamento.csv')
base = base.dropna()
base_treinamento = base.iloc[:, 2:6].values


normalizador = MinMaxScaler(feature_range=(0, 1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

previsores = []
preco_real = []

for i in range(150, len(base_treinamento[:,0])):
    previsores.append(base_treinamento_normalizada[i - 150:i, 0:4])
    preco_real.append(base_treinamento_normalizada[i, 0])
previsores, preco_real = np.array(previsores), np.array(preco_real)
#previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(previsores.shape[1], 4)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units=1, activation='sigmoid'))

regressor.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
# regressor.fit(previsores, preco_real, epochs=100, batch_size=32)
es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)
rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1)
mcp = ModelCheckpoint(filepath = 'pesos-{epoch:02d}.h5', monitor = 'loss',
                      save_best_only = True, verbose = 1)
regressor.fit(previsores, preco_real, epochs = 20, batch_size = 32,
              callbacks = [es, rlr, mcp])


# save the model to disk
#filename = 'eurusd_model.sav'
#pickle.dump(regressor, open(filename, 'wb'))

base_teste = pd.read_csv('EURUSD2019M1_teste.csv')
preco_real_teste = base_teste.iloc[:, 2:3].values
base_completa = pd.concat((base['<OPEN>'], base_teste['<OPEN>']), axis=0)
entradas = base_completa[len(base_completa) - len(base_teste) - 150:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(150, len(base_treinamento[:,0])):
    X_teste.append(entradas[i - 150:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

previsoes.mean()
preco_real_teste.mean()

plt.plot(preco_real_teste, color='red', label='Preço real')
plt.plot(previsoes, color='blue', label='Previsões')
plt.title('Previsão preço das ações')
plt.xlabel('Tempo')
plt.ylabel('Valores EURUSD')
plt.legend()
plt.show()