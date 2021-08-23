import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
sns.set(style="darkgrid", font_scale=1.5)
#Pre procesar
from sklearn.preprocessing import MinMaxScaler
#Modelo
import tensorflow as tf
from tensorflow.keras import layers



monedas = ["LTCUSDT", "BCHUSDT", "ETHUSDT", "BTCUSDT", "BNBUSDT"]

monedasName = {"LTCUSDT": "LiteCoin", "BCHUSDT": "Bitcoin Cash", "ETHUSDT": "Ethereum", "BTCUSDT": "Bitcoin", "BNBUSDT": "Binance Coin"}

for moneda in monedas:

#moneda = monedas[0]
    df = pd.read_csv(moneda + ".csv", parse_dates=["date"], index_col="date")

    df = df[["close"]]

    #Modelo con 360 timesteps desde t-360 a t-1. Usará "open" para hacer la predicción así que la dimensión del input es 1
    #Reorganizar la data de manera que para predecir un valor al tiempo t, tome en consideración los datos de hace 90 días. También es necesario normalizar los valores por el excesivo cálculo en una red neuronal

    #Pre procesamiento
    data = df.iloc[:, 0]
    hist = []
    target = []
    length = 90
    for i in range(len(data)-length):
        x = data[i:i+length]
        y = data[i+length]
        hist.append(x)
        target.append(y)

    hist = np.array(hist)
    target = np.array(target)

    target = target.reshape(-1, 1)

    sc = MinMaxScaler()
    hist_scaled = sc.fit_transform(hist)
    target_scaled = sc.fit_transform(target)
    hist_scaled = hist_scaled.reshape((len(hist_scaled), length, 1))
    dataLen = hist_scaled.shape[0]

    prediction = dataLen - 90 #Predecir tendencia de los siguientes 90 dias

    #Training and Test Sets
    X_train = hist_scaled[:prediction,:,:]
    X_test = hist_scaled[prediction:,:,:]
    y_train = target_scaled[:prediction,:]
    y_test = target_scaled[prediction:,:]
    model = tf.keras.Sequential()
    model.add(layers.LSTM(units=32, return_sequences=True,  #Primera capa oculta
                      input_shape=(90,1), dropout=0.2))
    model.add(layers.LSTM(units=32, return_sequences=True,
                      dropout=0.2))
    model.add(layers.LSTM(units=32, dropout=0.2))
    model.add(layers.Dense(units=1)) #Se usa para generar prediccion

    model.compile(optimizer='adam', loss='mean_squared_error')

    #batch size 32 y epocas 30

    model.fit(X_train, y_train, epochs=30, batch_size=1)
    if os.path.isfile("models/" + moneda + ".h5") is False:
        model.save('models' + moneda + ".h5")
'''
loss = history.history['loss']
epoch_count = range(1, len(loss) + 1)
plt.figure(figsize=(12,8))
plt.plot(epoch_count, loss, 'r--')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

#Prediccion
pred = model.predict(X_test)
pred_transformed = sc.inverse_transform(pred)
y_test_transformed = sc.inverse_transform(y_test)

plt.figure(figsize=(12,8))
plt.plot(y_test_transformed, color='blue', label='Real')
plt.plot(pred_transformed, color='red', label='Prediction')
plt.title(monedasName[moneda] + ' Price Prediction')
plt.legend()
plt.show()
'''