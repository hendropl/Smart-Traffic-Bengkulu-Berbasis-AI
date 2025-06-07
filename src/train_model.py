# 📁 src/train_model.py
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load data
df = pd.read_csv("../data/traffic_bengkulu.csv")

# Preprocess
scaler = MinMaxScaler()
scaled_volume = scaler.fit_transform(df[['volume']])

# Create sequences (24 jam sebelumnya untuk prediksi jam ke-25)
X, y = [], []
for i in range(24, len(scaled_volume)):
    X.append(scaled_volume[i-24:i])
    y.append(scaled_volume[i])

X, y = np.array(X), np.array(y)

# Build LSTM
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2,
          callbacks=[EarlyStopping(patience=3)], verbose=1)

# Save model and scaler
os.makedirs("../model", exist_ok=True)
model.save("../model/lstm_model.keras")
np.save("../model/scaler_minmax.npy", scaler.data_min_)
np.save("../model/scaler_maxabs.npy", scaler.data_max_)
print("✅ Model & scaler saved.")

# 📁 dashboard/app.py
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# Load model dan scaler
model = load_model("../model/lstm_model.h5", compile=False)
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = np.load("../model/scaler_minmax.npy"), 1 / (np.load("../model/scaler_maxabs.npy") - np.load("../model/scaler_minmax.npy"))

# Load data terbaru untuk prediksi
df = pd.read_csv("../data/traffic_bengkulu.csv")
data = df[['volume']].values[-24:]
data_scaled = scaler.transform(data).reshape(1, 24, 1)

# Streamlit UI
st.title("Prediksi Volume Lalu Lintas Bengkulu")

if st.button("Prediksi Volume Jam Berikutnya"):
    pred_scaled = model.predict(data_scaled)[0][0]
    pred_volume = scaler.inverse_transform([[pred_scaled]])[0][0]
    st.success(f"Prediksi volume jam berikutnya: {pred_volume:.0f} kendaraan/jam")
