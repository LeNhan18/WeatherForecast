import pandas as pd
import numpy as np
from tensorflow.python.keras.constraints import MinMaxNorm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.saving import save_model
#Đọc dữ lieu từ file csv
data = pd.read_csv("C:\\Users\\Admin\\PycharmProjects\\WeatherForecast\\Data\\weatherHistory.csv")
temps = data["Temperature (C)"].values.reshape(-1,1)

#Chuẩn hóa du lieu
scaler = MinMaxScaler()
temp_scaled = scaler.fit_transform(temps)

#Tao chuoi thời gian

def TaoChuoi(data, seq_length) :
    X, y = [], []
    for i in range(len(data)- seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)
seq_length = 7
X, y = TaoChuoi(temp_scaled, seq_length)

#Chia du lieu thanh train, test
train_size = int(len(X) * 0.8)
X_train, X_test =X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#Xây mô hinhf LSTM
model = Sequential()
model.add(LSTM(50, activation = 'relu',input_shape =(seq_length,1)))
model.add(Dense(1))
model.compile(optimizer ='adam',loss='mse')

#Huấn luyện mô hình
model.fit(X_train, y_train,epochs = 20,batch_size =32, validation_data =(X_test, y_test))
#Dự đoán
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

save_model(model ,"C:\\Users\\Admin\\PycharmProjects\\WeatherForecast\\Data\\WeatherForecast.keras")