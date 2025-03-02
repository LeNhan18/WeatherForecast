import pandas as pd
import numpy as np
from tensorflow.python.keras.constraints import MinMaxNorm
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from flask import Flask, render_template, jsonify
import plotly.express as px
app = Flask(__name__)
def TaoChuoi(data, seq_length) :
    X, y = [], []
    for i in range(len(data)- seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)
def Tai_va_HuanLuyen(model_path ="C:\\Users\\Admin\\PycharmProjects\\WeatherForecast\\Data\\WeatherForecast.keras",data_path="C:\\Users\\Admin\\PycharmProjects\\WeatherForecast\\Data\\weatherHistory.csv"):
    #Đọc dữ lieu từ file csv
    data = pd.read_csv(data_path)
    temps = data["Temperature (C)"].values.reshape(-1,1)

    #Chuẩn hóa du lieu
    scaler = MinMaxScaler()
    temp_scaled = scaler.fit_transform(temps)

    #Tao chuoi thời gian

    seq_length = 7
    X, y = TaoChuoi(temp_scaled, seq_length)

    #Chia du lieu thanh train, test
    train_size = int(len(X) * 0.8)
    X_train, X_test =X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # #Xây mô hinhf LSTM
    # model = Sequential()
    # model.add(LSTM(50, activation = 'relu',input_shape =(seq_length,1)))
    # model.add(Dense(1))
    # model.compile(optimizer ='adam',loss='mse')
    #
    # #Huấn luyện mô hình
    # model.fit(X_train, y_train,epochs = 20,batch_size =32, validation_data =(X_test, y_test))
    model = load_model(model_path)
    #Dự đoán
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions, y_test
# Tải dữ liệu và dự đoán
def load_and_predict(model_path="weather_model.keras", data_path="weatherHistory.csv"):
    try:
        # Đọc dữ liệu
        data = pd.read_csv(data_path)
        print("Cột trong dữ liệu:", data.columns.tolist())  # Debug: Kiểm tra cột
        # Đảm bảo cột nhiệt độ tồn tại (thay 'Temperature (C)' nếu cần)
        temps = data["Temperature (C)"].values.reshape(-1, 1)

        # Chuẩn hóa dữ liệu
        scaler = MinMaxScaler()
        temps_scaled = scaler.fit_transform(temps)

        # Tạo chuỗi thời gian
        seq_length = 7
        X, y = TaoChuoi(temps_scaled, seq_length)

        # Chia dữ liệu
        train_size = int(len(X) * 0.8)
        X_test = X[train_size:]
        y_test = y[train_size:]

        # Tải mô hình đã lưu
        model = load_model(model_path)
        predictions = model.predict(X_test, verbose=0)
        predictions = scaler.inverse_transform(predictions)
        y_test = scaler.inverse_transform(y_test)

        print("Dự đoán:", predictions[:5])  # Debug: Kiểm tra 5 dự đoán đầu
        print("Thực tế:", y_test[:5])      # Debug: Kiểm tra 5 giá trị thực tế đầu
        return predictions, y_test
    except Exception as e:
        print("Lỗi trong load_and_predict:", e)
        return None, None
def dudoan_ngaymai(model,scaler,recent_data,seq_length ):
    try:
        # Đảm bảo recent_data là mảng numpy với 7 giá trị gần nhất
        if len(recent_data) < seq_length:
            raise ValueError("Không đủ dữ liệu để dự đoán (cần ít nhất 7 giá trị).")

        # Chuẩn hóa dữ liệu gần nhất
        recent_data = np.array(recent_data).reshape(-1, 1)
        recent_scaled = scaler.transform(recent_data)

        # Tạo chuỗi đầu vào cho mô hình (shape: [1, seq_length, 1])
        input_seq = recent_scaled[-seq_length:].reshape(1, seq_length, 1)

        # Dự đoán
        prediction = model.predict(input_seq, verbose=0)
        prediction = scaler.inverse_transform(prediction)

        return float(round(prediction[0][0], 2))  # Trả về nhiệt độ dự đoán (°C)
    except Exception as e:
        print("Lỗi trong predict_tomorrow:", e)
        return None
@app.route('/')
def home():
    predictions, actual = Tai_va_HuanLuyen()
    if predictions is None or actual is None:
        return "Có lỗi xảy ra khi tải dữ liệu hoặc mô hình!", 500

    forecast_data = [
        {"day": i + 1, "predicted": round(pred[0], 2), "actual": round(act[0], 2)}
        for i, (pred, act) in enumerate(zip(predictions[:5], actual[:5]))
    ]
    return render_template('index.html', forecast_data=forecast_data)


# Route API trả về dự báo
@app.route('/api/forecast', methods=['GET'])
def forecast_api():
    predictions, actual = Tai_va_HuanLuyen()
    if predictions is None or actual is None:
        return jsonify({"error": "Có lỗi xảy ra khi tải dữ liệu hoặc mô hình"}), 500

    forecast_data = [
        {"day": i + 1, "predicted": float(round(pred[0], 2)), "actual": float(round(act[0], 2))}
        for i, (pred, act) in enumerate(zip(predictions[:5], actual[:5]))
    ]
    # Dự đoán nhiệt độ ngày mai
    return jsonify(forecast_data)
if __name__ == '__main__':
    app.run(debug=True)