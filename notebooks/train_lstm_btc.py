import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# Pastikan folder models tersedia
os.makedirs("models", exist_ok=True)

# 1. Memuat dan membersihkan data
df = pd.read_csv("dataset/coin_Bitcoin.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)
df = df[["Close"]].dropna()

# 2. Normalisasi Data (MinMaxScaler rentang 0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# 3. Membuat Sekuens (Windowing) dengan timesteps = 60
timesteps = 60
X, y = [], []

for i in range(timesteps, len(scaled_data)):
  X.append(scaled_data[i - timesteps : i, 0])
  y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
# Reshape ke format tensor 3D: (samples, timesteps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. Pembagian Data Latih (80%) dan Data Uji (20%) secara kronologis
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. Membangun Arsitektur Model LSTM sesuai Modul 04
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1),
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# 6. Melatih Model
history = model.fit(
    X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test)
)

# 7. Evaluasi Kinerja Model pada Data Uji
predictions = model.predict(X_test)
predictions_actual = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
mae = mean_absolute_error(y_test_actual, predictions_actual)
print(f"RMSE: ${rmse:.2f} | MAE: ${mae:.2f}")

# 8. Simpan Model, Scaler, dan Hasil Evaluasi untuk Aplikasi Web Flask
model.save("models/lstm_btc_model.h5")
joblib.dump(scaler, "models/scaler_btc.pkl")

# Simpan data hasil uji untuk ditampilkan di dashboard web
test_dates = df.index[timesteps + train_size :].strftime("%Y-%m-%d").tolist()
eval_data = {
    "dates": test_dates[-120:],  # 120 hari terakhir data uji
    "actual": y_test_actual.flatten()[-120:].tolist(),
    "predicted": predictions_actual.flatten()[-120:].tolist(),
    "rmse": round(float(rmse), 2),
    "mae": round(float(mae), 2),
    "last_60_days": df["Close"].values[-60:].tolist(),
    "last_date": df.index[-1].strftime("%Y-%m-%d"),
}
joblib.dump(eval_data, "models/eval_data.pkl")
print("Semua artefak model berhasil disimpan ke folder models/!")