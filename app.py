import os
from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Memuat model, scaler, dan data evaluasi
MODEL_PATH = os.path.join("models", "lstm_btc_model.h5")
SCALER_PATH = os.path.join("models", "scaler_btc.pkl")
EVAL_PATH = os.path.join("models", "eval_data.pkl")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
eval_data = joblib.load(EVAL_PATH)


@app.route("/")
def index():
  return render_template(
      "index.html",
      rmse=eval_data["rmse"],
      mae=eval_data["mae"],
      last_date=eval_data["last_date"],
      last_price=round(eval_data["last_60_days"][-1], 2),
  )


@app.route("/api/history")
def get_history():
  return jsonify({
      "dates": eval_data["dates"],
      "actual": [round(val, 2) for val in eval_data["actual"]],
      "predicted": [round(val, 2) for val in eval_data["predicted"]],
  })


@app.route("/api/forecast", methods=["POST"])
def forecast_future():
  data = request.get_json()
  days_ahead = int(data.get("days", 7))
  days_ahead = max(1, min(days_ahead, 30))  # Batasi 1 - 30 hari

  # Ambil 60 hari terakhir dari dataset sebagai input awal
  current_sequence = np.array(eval_data["last_60_days"]).reshape(-1, 1)
  scaled_seq = scaler.transform(current_sequence)

  future_predictions = []
  input_seq = list(scaled_seq.flatten())

  # Prediksi rekursif hari demi hari
  for _ in range(days_ahead):
    x_input = np.array(input_seq[-60:]).reshape(1, 60, 1)
    pred_scaled = model.predict(x_input, verbose=0)[0][0]
    input_seq.append(pred_scaled)
    pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
    future_predictions.append(round(float(pred_price), 2))

  return jsonify({
      "days": days_ahead,
      "predictions": future_predictions,
      "next_day_price": future_predictions[0],
  })


if __name__ == "__main__":
  app.run(debug=True)