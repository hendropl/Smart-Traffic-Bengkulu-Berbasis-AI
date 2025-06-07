# enhanced_model.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

class EnhancedTrafficPredictor:
    def __init__(self):
        self.model = None
        self.volume_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.weather_encoder = LabelEncoder()
        self.day_encoder = LabelEncoder()
        self.sequence_length = 24
        
    def create_enhanced_features(self, df):
        """
        Membuat fitur tambahan untuk meningkatkan akurasi prediksi
        """
        # Pastikan timestamp dalam format datetime
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Fitur waktu
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
        
        # Fitur cuaca (simulasi jika tidak ada data real)
        if 'weather' not in df.columns:
            weather_conditions = ['sunny', 'rainy', 'cloudy', 'partly_cloudy']
            df['weather'] = np.random.choice(weather_conditions, len(df))
        
        if 'temperature' not in df.columns:
            df['temperature'] = np.random.normal(28, 3, len(df))  # Suhu rata-rata Bengkulu
        
        if 'humidity' not in df.columns:
            df['humidity'] = np.random.normal(75, 10, len(df))  # Kelembaban rata-rata
        
        # Fitur event/holiday (simulasi)
        if 'is_holiday' not in df.columns:
            df['is_holiday'] = np.random.choice([0, 1], len(df), p=[0.95, 0.05])
        
        # Fitur lag (volume sebelumnya)
        df['volume_lag_1'] = df['volume'].shift(1)
        df['volume_lag_2'] = df['volume'].shift(2)
        df['volume_lag_24'] = df['volume'].shift(24)  # Volume hari sebelumnya jam yang sama
        
        # Rolling statistics
        df['volume_rolling_mean_3'] = df['volume'].rolling(window=3).mean()
        df['volume_rolling_std_3'] = df['volume'].rolling(window=3).std()
        df['volume_rolling_mean_24'] = df['volume'].rolling(window=24).mean()
        
        # Cyclical encoding untuk fitur waktu
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def prepare_enhanced_data(self, df):
        """
        Menyiapkan data dengan fitur tambahan untuk training
        """
        # Buat fitur tambahan
        df_enhanced = self.create_enhanced_features(df)
        
        # Drop missing values
        df_enhanced = df_enhanced.dropna()
        
        # Encode categorical variables
        df_enhanced['weather_encoded'] = self.weather_encoder.fit_transform(df_enhanced['weather'])
        
        # Pilih fitur untuk training
        feature_columns = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'is_weekend', 'is_rush_hour', 'is_holiday',
            'temperature', 'humidity', 'weather_encoded',
            'volume_lag_1', 'volume_lag_2', 'volume_lag_24',
            'volume_rolling_mean_3', 'volume_rolling_std_3', 'volume_rolling_mean_24'
        ]
        
        # Pastikan semua kolom ada
        for col in feature_columns:
            if col not in df_enhanced.columns:
                df_enhanced[col] = 0
        
        # Scale fitur
        features = df_enhanced[feature_columns].values
        volume = df_enhanced['volume'].values.reshape(-1, 1)
        
        # Scale data
        features_scaled = self.feature_scaler.fit_transform(features)
        volume_scaled = self.volume_scaler.fit_transform(volume)
        
        return features_scaled, volume_scaled.flatten(), df_enhanced
    
    def create_sequences(self, features, volume):
        """
        Membuat sequences untuk LSTM dengan fitur tambahan
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(volume)):
            # Gabungkan volume dan fitur untuk sequence
            volume_seq = volume[i-self.sequence_length:i]
            feature_seq = features[i-self.sequence_length:i]
            
            # Gabungkan volume dengan fitur
            combined_seq = np.column_stack([volume_seq.reshape(-1, 1), feature_seq])
            
            X.append(combined_seq)
            y.append(volume[i])
        
        return np.array(X), np.array(y)
    
    def build_enhanced_model(self, input_shape):
        """
        Membangun model LSTM yang lebih kompleks dengan fitur tambahan
        """
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            Dense(50, activation='relu'),
            Dropout(0.1),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_enhanced_model(self, df, epochs=100, batch_size=32, validation_split=0.2):
        """
        Melatih model dengan fitur tambahan
        """
        print("Menyiapkan data dengan fitur tambahan...")
        features_scaled, volume_scaled, df_enhanced = self.prepare_enhanced_data(df)
        
        print("Membuat sequences...")
        X, y = self.create_sequences(features_scaled, volume_scaled)
        
        print(f"Shape data: X={X.shape}, y={y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Build model
        self.model = self.build_enhanced_model((X.shape[1], X.shape[2]))
        
        print("Melatih model...")
        print(self.model.summary())
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.0001
        )
        
        # Training
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluasi
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        return history, df_enhanced
    
    def predict_with_features(self, recent_data, current_features):
        """
        Prediksi dengan menggunakan fitur tambahan
        """
        if self.model is None:
            raise ValueError("Model belum dilatih!")
        
        # Persiapkan data untuk prediksi
        recent_volume_scaled = self.volume_scaler.transform(recent_data.reshape(-1, 1)).flatten()
        current_features_scaled = self.feature_scaler.transform([current_features])
        
        # Buat sequence
        combined_seq = np.column_stack([
            recent_volume_scaled.reshape(-1, 1),
            np.repeat(current_features_scaled, len(recent_volume_scaled), axis=0)
        ])
        
        X_pred = combined_seq.reshape(1, self.sequence_length, -1)
        
        # Prediksi
        pred_scaled = self.model.predict(X_pred, verbose=0)[0][0]
        pred_volume = self.volume_scaler.inverse_transform([[pred_scaled]])[0][0]
        
        return pred_volume
    
    def predict_multiple_hours(self, recent_data, current_features, hours=6):
        """
        Prediksi untuk beberapa jam ke depan
        """
        predictions = []
        current_data = recent_data.copy()
        
        for hour in range(hours):
            # Update fitur waktu untuk jam berikutnya
            next_features = current_features.copy()
            current_hour = int(next_features[0] * 24)  # Asumsi fitur pertama adalah hour_sin
            next_hour = (current_hour + hour + 1) % 24
            
            # Update cyclical encoding untuk jam
            next_features[0] = np.sin(2 * np.pi * next_hour / 24)  # hour_sin
            next_features[1] = np.cos(2 * np.pi * next_hour / 24)  # hour_cos
            
            # Prediksi
            pred = self.predict_with_features(current_data, next_features)
            predictions.append(pred)
            
            # Update data untuk prediksi selanjutnya
            current_data = np.roll(current_data, -1)
            current_data[-1] = pred
        
        return predictions
    
    def save_model(self, model_path="../model/", prefix="enhanced_"):
        """
        Simpan model dan scaler
        """
        if self.model is None:
            raise ValueError("Model belum dilatih!")
        
        # Simpan model
        self.model.save(f"{model_path}{prefix}lstm_model.keras")
        
        # Simpan scalers
        with open(f"{model_path}{prefix}volume_scaler.pkl", 'wb') as f:
            pickle.dump(self.volume_scaler, f)
        
        with open(f"{model_path}{prefix}feature_scaler.pkl", 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        with open(f"{model_path}{prefix}weather_encoder.pkl", 'wb') as f:
            pickle.dump(self.weather_encoder, f)
        
        print(f"Model dan scaler berhasil disimpan di {model_path}")
    
    def load_model(self, model_path="../model/", prefix="enhanced_"):
        """
        Load model dan scaler
        """
        from tensorflow.keras.models import load_model
        
        # Load model
        self.model = load_model(f"{model_path}{prefix}lstm_model.keras")
        
        # Load scalers
        with open(f"{model_path}{prefix}volume_scaler.pkl", 'rb') as f:
            self.volume_scaler = pickle.load(f)
        
        with open(f"{model_path}{prefix}feature_scaler.pkl", 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        with open(f"{model_path}{prefix}weather_encoder.pkl", 'rb') as f:
            self.weather_encoder = pickle.load(f)
        
        print("Model berhasil dimuat!")

# Fungsi untuk congestion detection dan alert system
class CongestionDetector:
    def __init__(self):
        self.thresholds = {
            'very_high': 800,
            'high': 600,
            'medium': 400,
            'low': 200
        }
        self.alert_history = []
    
    def detect_congestion_level(self, volume):
        """
        Deteksi tingkat kemacetan berdasarkan volume
        """
        if volume >= self.thresholds['very_high']:
            return 'very_high', '🔴', 'Sangat Padat'
        elif volume >= self.thresholds['high']:
            return 'high', '🟠', 'Padat'
        elif volume >= self.thresholds['medium']:
            return 'medium', '🟡', 'Sedang'
        elif volume >= self.thresholds['low']:
            return 'low', '🟢', 'Lancar'
        else:
            return 'very_low', '🔵', 'Sangat Lancar'
    
    def generate_alert(self, current_volume, predicted_volume, location="Default"):
        """
        Generate alert berdasarkan kondisi saat ini dan prediksi
        """
        current_level, current_icon, current_desc = self.detect_congestion_level(current_volume)
        pred_level, pred_icon, pred_desc = self.detect_congestion_level(predicted_volume)
        
        alert = {
            'timestamp': pd.Timestamp.now(),
            'location': location,
            'current_volume': current_volume,
            'predicted_volume': predicted_volume,
            'current_level': current_level,
            'predicted_level': pred_level,
            'alert_type': None,
            'message': '',
            'recommendations': []
        }
        
        # Determine alert type
        if pred_level == 'very_high':
            alert['alert_type'] = 'critical'
            alert['message'] = f"🚨 PERINGATAN KRITIS: Kemacetan parah diprediksi di {location}!"
        elif pred_level == 'high' and current_level in ['medium', 'low', 'very_low']:
            alert['alert_type'] = 'warning'
            alert['message'] = f"⚠️ PERINGATAN: Kemacetan akan meningkat di {location}"
        elif current_level == 'very_high':
            alert['alert_type'] = 'current_critical'
            alert['message'] = f"🔴 KEMACETAN PARAH: Kondisi lalu lintas sangat buruk di {location}"
        
        # Generate recommendations
        alert['recommendations'] = self.get_recommendations(pred_level, location)
        
        # Simpan ke history
        self.alert_history.append(alert)
        
        return alert
    
    def get_recommendations(self, congestion_level, location):
        """
        Berikan rekomendasi berdasarkan tingkat kemacetan
        """
        recommendations = {
            'very_high': [
                "🚨 Hindari area ini jika memungkinkan",
                "🕐 Tunda perjalanan 1-2 jam jika tidak mendesak",
                "🚌 Gunakan transportasi umum alternatif",
                "📱 Pantau update real-time secara berkala"
            ],
            'high': [
                "⏰ Tambahkan 20-30 menit waktu perjalanan",
                "🛣️ Pertimbangkan rute alternatif",
                "🚗 Berkendara dengan sabar",
                "📍 Cek kondisi lalu lintas sebelum berangkat"
            ],
            'medium': [
                "✅ Kondisi normal, tetap waspada",
                "🛣️ Rute utama dapat dilalui",
                "⏱️ Waktu perjalanan sesuai estimasi normal"
            ],
            'low': [
                "🟢 Kondisi lalu lintas baik",
                "🛣️ Semua rute dapat dilalui lancar",
                "⚡ Waktu perjalanan optimal"
            ],
            'very_low': [
                "🔵 Kondisi ideal untuk berkendara",
                "🎯 Pilih rute tercepat yang tersedia",
                "⏰ Waktu perjalanan minimal"
            ]
        }
        
        return recommendations.get(congestion_level, ["Pantau kondisi lalu lintas"])

# Route optimization system
class RouteOptimizer:
    def __init__(self):
        # Data rute di Bengkulu (simulasi)
        self.routes_data = {
            ('Terminal Panorama', 'Pasar Minggu'): [
                {'name': 'Jl. Suprapto - Jl. Ahmad Yani', 'distance': 5.2, 'base_time': 12, 'traffic_factor': 1.2},
                {'name': 'Jl. Veteran - Jl. Pariwisata', 'distance': 6.1, 'base_time': 15, 'traffic_factor': 0.9},
                {'name': 'Jl. Basuki Rahmat - Jl. Panorama', 'distance': 7.3, 'base_time': 18, 'traffic_factor': 0.8}
            ],
            ('Universitas Bengkulu', 'Bandara Fatmawati'): [
                {'name': 'Jl. Raya Kandang - Jl. Bandara', 'distance': 15.4, 'base_time': 25, 'traffic_factor': 1.1},
                {'name': 'Jl. Lintas Sumatera', 'distance': 18.2, 'base_time': 30, 'traffic_factor': 0.9},
                {'name': 'Jl. Pantai Panjang - Jl. Zainul Arifin', 'distance': 16.8, 'base_time': 28, 'traffic_factor': 1.0}
            ]
        }
    
    def optimize_route(self, origin, destination, current_traffic_data):
        """
        Optimasi rute berdasarkan kondisi lalu lintas real-time
        """
        route_key = (origin, destination)
        reverse_key = (destination, origin)
        
        # Cari rute yang tersedia
        available_routes = None
        if route_key in self.routes_data:
            available_routes = self.routes_data[route_key]
        elif reverse_key in self.routes_data:
            available_routes = self.routes_data[reverse_key]
        else:
            # Generate rute default jika tidak ada data
            available_routes = [
                {'name': 'Rute Utama', 'distance': 10.0, 'base_time': 20, 'traffic_factor': 1.2},
                {'name': 'Rute Alternatif 1', 'distance': 12.0, 'base_time': 18, 'traffic_factor': 0.9},
                {'name': 'Rute Alternatif 2', 'distance': 8.5, 'base_time': 25, 'traffic_factor': 1.1}
            ]
        
        # Hitung waktu tempuh dengan kondisi lalu lintas
        optimized_routes = []
        for route in available_routes:
            # Simulasi pengaruh lalu lintas terhadap waktu tempuh
            traffic_multiplier = route['traffic_factor'] * (1 + current_traffic_data.get('congestion_factor', 0))
            estimated_time = route['base_time'] * traffic_multiplier
            
            # Tentukan status lalu lintas
            if traffic_multiplier >= 1.5:
                traffic_status = 'Padat'
                status_icon = '🔴'
            elif traffic_multiplier >= 1.2:
                traffic_status = 'Sedang'
                status_icon = '🟡'
            else:
                traffic_status = 'Lancar'
                status_icon = '🟢'
            
            optimized_routes.append({
                'name': route['name'],
                'distance': route['distance'],
                'estimated_time': estimated_time,
                'traffic_status': traffic_status,
                'status_icon': status_icon,
                'traffic_multiplier': traffic_multiplier
            })
        
        # Sort berdasarkan waktu tempuh
        optimized_routes.sort(key=lambda x: x['estimated_time'])
        
        return optimized_routes

# Example usage dan training script
if __name__ == "__main__":
    # Contoh penggunaan
    print("=== Enhanced Traffic Prediction System ===")
    
    # Load atau generate sample data
    try:
        df = pd.read_csv("../data/traffic_bengkulu.csv")
        print(f"Data loaded: {len(df)} records")
    except:
        print("Generating sample data...")
        # Generate sample data jika file tidak ada
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
        volumes = []
        
        for date in dates:
            # Simulasi pola lalu lintas realistis
            hour = date.hour
            day_of_week = date.dayofweek
            
            # Base volume berdasarkan jam
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hour
                base_volume = np.random.randint(600, 900)
            elif 10 <= hour <= 16:  # Siang hari
                base_volume = np.random.randint(400, 700)
            elif 20 <= hour <= 23:  # Malam
                base_volume = np.random.randint(300, 500)
            else:  # Dini hari
                base_volume = np.random.randint(100, 300)
            
            # Pengaruh hari
            if day_of_week >= 5:  # Weekend
                base_volume *= 0.8
            
            # Tambahkan noise
            volume = base_volume + np.random.randint(-50, 50)
            volume = max(50, volume)  # Minimum volume
            
            volumes.append(volume)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'volume': volumes
        })
        
        print(f"Sample data generated: {len(df)} records")
    
    # Initialize predictor
    predictor = EnhancedTrafficPredictor()
    
    # Train model (uncomment untuk training)
    print("\nTraining enhanced model...")
    history, df_enhanced = predictor.train_enhanced_model(df, epochs=50)
    predictor.save_model()
    
    # Initialize congestion detector
    detector = CongestionDetector()
    
    # Initialize route optimizer
    optimizer = RouteOptimizer()
    
    print("\nSistem Enhanced Traffic Prediction siap digunakan!")