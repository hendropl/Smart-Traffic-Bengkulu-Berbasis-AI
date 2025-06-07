import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')
import requests
# Konfigurasi halaman
st.set_page_config(
    page_title="Smart Traffic Bengkulu AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state untuk mencegah kedip
if 'traffic_data' not in st.session_state:
    st.session_state.traffic_data = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'training_message' not in st.session_state:
    st.session_state.training_message = ""

# CSS untuk styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .ai-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        border: 2px solid #00d4ff;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { box-shadow: 0 0 5px #00d4ff; }
        to { box-shadow: 0 0 20px #00d4ff, 0 0 30px #00d4ff; }
    }
    .alert-high {
        background-color: #ff4757;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .alert-medium {
        background-color: #ffa502;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .alert-low {
        background-color: #2ed573;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .ai-info {
        background: linear-gradient(45deg, #1e3c72, #2a5298);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00d4ff;
        color: white;
        margin: 1rem 0;
    }
    .route-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #00d4ff;
    }
    .alternative-route {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 5px solid #ff6b6b;
    }
    .best-route {
        background: linear-gradient(135deg, #2ed573 0%, #17a2b8 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border: 3px solid #00ff00;
        animation: pulse 2s ease-in-out infinite alternate;
    }
    @keyframes pulse {
        from { transform: scale(1); }
        to { transform: scale(1.02); }
    }
</style>
""", unsafe_allow_html=True)

class TrafficLSTMPredictor:
    def __init__(self, sequence_length=12):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        
    def create_sequences(self, data, target_col='volume'):
        """Membuat sekuens data untuk LSTM"""
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(data)):
            sequences.append(data.iloc[i-self.sequence_length:i][target_col].values)
            targets.append(data.iloc[i][target_col])
            
        return np.array(sequences), np.array(targets)
    
    def build_model(self, input_shape):
        """Membangun model LSTM"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def generate_synthetic_data(self, days=30, seed=42):
        """Generate data sintetis yang realistis untuk training dengan seed tetap"""
        np.random.seed(seed)  # Seed tetap untuk hasil konsisten
        
        # Parameter untuk pola lalu lintas
        base_volume = 400
        hours = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='h')
        
        volumes = []
        for hour in hours:
            # Pola harian (jam sibuk pagi dan sore)
            hour_of_day = hour.hour
            if 7 <= hour_of_day <= 9:  # Rush hour pagi
                daily_factor = 1.8
            elif 17 <= hour_of_day <= 19:  # Rush hour sore
                daily_factor = 2.0
            elif 12 <= hour_of_day <= 14:  # Jam makan siang
                daily_factor = 1.3
            elif 22 <= hour_of_day or hour_of_day <= 5:  # Malam/dini hari
                daily_factor = 0.3
            else:
                daily_factor = 1.0
            
            # Pola mingguan (weekend lebih sepi)
            if hour.weekday() >= 5:  # Weekend
                weekly_factor = 0.7
            else:
                weekly_factor = 1.0
            
            # Noise random
            noise = np.random.normal(0, 50)
            
            # Trend jangka panjang
            trend = np.sin(hour.dayofyear / 365 * 2 * np.pi) * 50
            
            volume = base_volume * daily_factor * weekly_factor + noise + trend
            volume = max(50, min(900, volume))  # Batas volume
            volumes.append(volume)
        
        return pd.DataFrame({
            'datetime': hours,
            'volume': volumes,
            'hour': [h.hour for h in hours],
            'day_of_week': [h.weekday() for h in hours],
            'is_weekend': [h.weekday() >= 5 for h in hours]
        })
    
    def train_model(self, route_key="default"):
        """Training model LSTM dengan data sintetis"""
        try:
            # Generate data training
            data = self.generate_synthetic_data(days=30)
            
            # Normalisasi data
            data_scaled = data.copy()
            data_scaled['volume'] = self.scaler.fit_transform(data[['volume']])
            
            # Buat sequences
            X, y = self.create_sequences(data_scaled)
            
            if len(X) == 0:
                return False, "Data tidak cukup untuk training"
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Reshape untuk LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Build dan train model
            self.model = self.build_model((X_train.shape[1], 1))
            
            # Training dengan early stopping
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Evaluasi model
            y_pred = self.model.predict(X_test, verbose=0)
            y_pred_original = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            mae = mean_absolute_error(y_test_original, y_pred_original)
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
            
            self.is_trained = True
            self.training_history = history.history
            self.mae = mae
            self.rmse = rmse
            
            return True, f"Model berhasil dilatih. MAE: {mae:.2f}, RMSE: {rmse:.2f}"
            
        except Exception as e:
            return False, f"Error training model: {str(e)}"
    
    def predict_next_hours(self, current_data, hours_ahead=3):
        """Prediksi untuk beberapa jam ke depan"""
        if not self.is_trained or self.model is None:
            return None, "Model belum dilatih"
        
        try:
            # Prepare data
            if len(current_data) < self.sequence_length:
                # Pad dengan data rata-rata jika kurang
                padding = [np.mean(current_data)] * (self.sequence_length - len(current_data))
                current_data = padding + list(current_data)
            
            # Ambil sequence terbaru
            sequence = current_data[-self.sequence_length:]
            
            # Normalisasi
            sequence_scaled = self.scaler.transform(np.array(sequence).reshape(-1, 1)).flatten()
            
            predictions = []
            current_sequence = sequence_scaled.copy()
            
            for _ in range(hours_ahead):
                # Reshape untuk prediksi
                X_pred = current_sequence.reshape((1, self.sequence_length, 1))
                
                # Prediksi
                pred_scaled = self.model.predict(X_pred, verbose=0)[0][0]
                pred_original = self.scaler.inverse_transform([[pred_scaled]])[0][0]
                
                predictions.append(max(50, min(900, pred_original)))
                
                # Update sequence untuk prediksi berikutnya
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = pred_scaled
            
            return predictions, "Sukses"
            
        except Exception as e:
            return None, f"Error dalam prediksi: {str(e)}"

# Cache untuk model predictor dengan hash yang stabil
@st.cache_resource
def get_lstm_predictor():
    """Get cached LSTM predictor"""
    predictor = TrafficLSTMPredictor()
    return predictor

def classify_traffic_level(volume):
    """Klasifikasi tingkat kemacetan"""
    if volume >= 800:
        return "Sangat Padat", "🔴"
    elif volume >= 600:
        return "Padat", "🟠"
    elif volume >= 400:
        return "Sedang", "🟡"
    elif volume >= 200:
        return "Lancar", "🟢"
    else:
        return "Sangat Lancar", "🔵"

def get_traffic_recommendations(level, volume, ai_confidence=None):
    """Berikan rekomendasi berdasarkan tingkat kemacetan dengan AI insights"""
    base_recommendations = {
        "Sangat Padat": [
            "🚨 AI merekomendasikan hindari rute utama",
            "⏰ Prediksi: Tunda perjalanan 1-2 jam untuk efisiensi optimal",
            "🚌 Algoritma menyarankan transportasi umum",
            "🛣️ Analisis AI: Rute alternatif 70% lebih cepat"
        ],
        "Padat": [
            "⚠️ AI prediksi: Siapkan waktu tambahan 15-30 menit",
            "🛣️ Machine learning merekomendasikan rute alternatif",
            "🚗 Neural network: Berkendara dengan kecepatan optimal",
            "📱 AI monitoring: Pantau update real-time"
        ],
        "Sedang": [
            "✅ AI analysis: Kondisi normal terdeteksi",
            "🛣️ Deep learning: Rute utama aman dilalui",
            "⏱️ Prediksi akurat: Waktu perjalanan sesuai estimasi",
            "🚦 AI reminder: Patuhi protokol lalu lintas"
        ],
        "Lancar": [
            "🟢 AI confirmation: Kondisi lalu lintas optimal",
            "🛣️ Machine learning: Semua rute dalam kondisi baik",
            "⚡ Neural network: Waktu perjalanan efisien",
            "🚗 AI safety: Tetap berkendara aman"
        ],
        "Sangat Lancar": [
            "🔵 AI verdict: Kondisi ideal terdeteksi",
            "🛣️ Deep learning: Pilih rute optimal tersedia",
            "⏰ Prediksi AI: Waktu perjalanan minimal",
            "🎯 Machine learning: Optimal untuk perjalanan jauh"
        ]
    }
    
    recommendations = base_recommendations.get(level, ["AI monitoring sistem aktif"])
    
    if ai_confidence is not None:
        confidence_text = f"🤖 Tingkat kepercayaan AI: {ai_confidence:.1f}%"
        recommendations.append(confidence_text)
    
    return recommendations

def get_bengkulu_locations():
    """Koordinat tetap hasil lookup dari OSM (bukan Google Maps)"""
    return {
        "Terminal Panorama": {"lat": -3.80966, "lng": 102.29404},
        "Pasar Minggu": {"lat": -3.793484, "lng": 102.266439},
        "Universitas Bengkulu": {"lat": -3.75874, "lng":  102.27139},
        "Bandar Udara Fatmawati Soekarno": {"lat": -3.8629249, "lng": 102.3393898},
        "Pelabuhan Pulau Baai": {"lat": -3.90492, "lng": 102.30666},
        "Masjid Jamik": {"lat": -3.792397, "lng":  102.262121},
        "Alun-alun Bengkulu": {"lat": -3.7634, "lng":  102.26815},
        "Mall Bengkulu Indah": {"lat": -3.81210, "lng":  102.26815},
        "Stadion Semarak": {"lat": -3.793453, "lng":  102.272844},
        "Pantai Panjang": {"lat": -3.804029, "lng":  102.257652},
        "Benteng Marlborough": {"lat": -3.787185, "lng": 102.251816},
        "UIN Fatmawati Sukarno Bengkulu": {"lat": -3.83466, "lng":  102.32799},
        "Rumah Sakit Bhayangkara Tingkat IV": {"lat": -3.7900511, "lng": 102.2504825},
        "Jl. Veteran": {"lat": -3.791488, "lng": 102.252814},
        "Jl. Ratu Agung": {"lat": -3.8001227, "lng": 102.2573724},
        "Jl. Basuki Rahmat": {"lat": -3.795646,  "lng": 102.267941},
        "Jl. Pariwisata": {"lat":-3.800763, "lng":  102.254895},
        "Jl. Rawa Makmur": {"lat": -3.77986, "lng": 102.27430},
        "Jl. Salak Raya": {"lat": -3.82061, "lng":  102.30555},
        "Jl. Danau": {"lat": -3.80798,  "lng": 102.30074},
        "Komplek Unib": {"lat": -3.7473204, "lng": 102.2918568},
        "Padang Kemiling": {"lat": -3.85490, "lng": 102.33572},
        "Muara Bangkahulu": {"lat":-3.76335, "lng":   102.28544},
        "Gading Cempaka": {"lat": -3.7650414, "lng": 102.2899335},
        "Soeprapto": {"lat": -3.795143, "lng":  102.264251},
        "Teluk Segara": {"lat": -3.7916853, "lng": 102.2538002},
        "Pematang Gubernur": {"lat": -3.76259, "lng":  102.29297},
        "Sawah Lebar": {"lat": -3.794190, "lng":  102.276149}
    }


def calculate_distance(coord1, coord2):
    """Hitung jarak antara dua koordinat (simplified)"""
    lat_diff = abs(coord1['lat'] - coord2['lat'])
    lng_diff = abs(coord1['lng'] - coord2['lng'])
    return np.sqrt(lat_diff**2 + lng_diff**2) * 111  # Rough conversion to km

def find_intermediate_points(origin, destination, locations):
    """Cari titik-titik intermediate untuk rute alternatif"""
    origin_coord = locations[origin]
    dest_coord = locations[destination]
    
    # Hitung titik tengah
    mid_lat = (origin_coord['lat'] + dest_coord['lat']) / 2
    mid_lng = (origin_coord['lng'] + dest_coord['lng']) / 2
    
    # Cari lokasi yang dekat dengan titik tengah
    intermediate_points = []
    for name, coord in locations.items():
        if name not in [origin, destination]:
            dist_to_mid = calculate_distance({'lat': mid_lat, 'lng': mid_lng}, coord)
            dist_to_origin = calculate_distance(origin_coord, coord)
            dist_to_dest = calculate_distance(coord, dest_coord)
            
            # Pilih lokasi yang membentuk rute yang masuk akal
            if dist_to_mid < 5 and dist_to_origin > 1 and dist_to_dest > 1:
                intermediate_points.append({
                    'name': name,
                    'coord': coord,
                    'distance_factor': dist_to_origin + dist_to_dest
                })
    
    # Sort berdasarkan efisiensi rute
    intermediate_points.sort(key=lambda x: x['distance_factor'])
    
    return intermediate_points[:5]  # Ambil 5 teratas

def get_stable_route_data(origin, destination, locations, seed_base=42):
    """Generate data rute yang stabil berdasarkan origin-destination"""
    # Buat seed yang konsisten berdasarkan origin dan destination
    seed = hash(f"{origin}-{destination}") % 1000000 + seed_base
    np.random.seed(seed)
    
    origin_coord = locations[origin]
    dest_coord = locations[destination]
    
    # Rute utama (direct)
    main_route = {
        "name": "Rute Utama",
        "type": "main",
        "waypoints": [origin_coord, dest_coord],
        "waypoint_names": [origin, destination],
        "distance": calculate_distance(origin_coord, dest_coord),
        "traffic_volume": np.random.randint(400, 900),
        "road_type": "Jalan Utama",
        "description": f"Rute langsung dari {origin} ke {destination}",
        "estimated_time": None,
        "congestion_level": None
    }
    
    # Cari intermediate points untuk rute alternatif
    intermediate_points = find_intermediate_points(origin, destination, locations)
    
    routes = [main_route]
    
    # Buat rute alternatif berdasarkan intermediate points
    for i, point in enumerate(intermediate_points[:3]):  # Ambil 3 rute alternatif
        waypoint_name = point['name']
        waypoint_coord = point['coord']
        
        alt_route = {
            "name": f"Rute Alternatif {i+1}",
            "type": "alternative",
            "waypoints": [origin_coord, waypoint_coord, dest_coord],
            "waypoint_names": [origin, waypoint_name, destination],
            "distance": calculate_distance(origin_coord, waypoint_coord) + calculate_distance(waypoint_coord, dest_coord),
            "traffic_volume": np.random.randint(150, 600),
            "road_type": "Jalan Alternatif",
            "description": f"Via {waypoint_name} - {get_route_description(i)}",
            "estimated_time": None,
            "congestion_level": None
        }
        routes.append(alt_route)
    
    # Hitung estimasi waktu dan tingkat kemacetan untuk setiap rute
    for route in routes:
        route['estimated_time'] = calculate_travel_time(route['distance'], route['traffic_volume'])
        route['congestion_level'], route['congestion_icon'] = classify_traffic_level(route['traffic_volume'])
    
    return routes

def get_route_description(index):
    """Deskripsi rute alternatif"""
    descriptions = [
        "Rute memutar tapi lebih lancar",
        "Jalur alternatif dengan pemandangan",
        "Rute pintas melalui jalan dalam kota",
        "Jalur lingkar dengan lalu lintas minimal",
        "Rute alternatif dengan fasilitas lengkap"
    ]
    return descriptions[index % len(descriptions)]

def calculate_travel_time(distance, traffic_volume):
    """Hitung estimasi waktu perjalanan berdasarkan jarak dan volume lalu lintas"""
    if traffic_volume >= 800:
        avg_speed = 15  # km/h (sangat lambat)
    elif traffic_volume >= 600:
        avg_speed = 25  # km/h (lambat)
    elif traffic_volume >= 400:
        avg_speed = 40  # km/h (sedang)
    elif traffic_volume >= 200:
        avg_speed = 50  # km/h (lancar)
    else:
        avg_speed = 60  # km/h (sangat lancar)
    
    time_hours = distance / avg_speed
    time_minutes = time_hours * 60
    
    return time_minutes

def get_route_recommendation(routes, predictor):
    """AI recommendation untuk rute terbaik berdasarkan prediksi"""
    best_route = None
    best_score = -1
    
    for route in routes:
        current_volume = route['traffic_volume']
        
        if predictor.is_trained:
            historical_data = [current_volume + np.random.randint(-50, 50) for _ in range(12)]
            predictions, _ = predictor.predict_next_hours(historical_data, 3)
            
            if predictions:
                avg_predicted_volume = np.mean(predictions)
                prediction_score = max(0, (900 - avg_predicted_volume) / 900 * 100)
            else:
                prediction_score = max(0, (900 - current_volume) / 900 * 100)
        else:
            prediction_score = max(0, (900 - current_volume) / 900 * 100)
        
        distance_score = max(0, (50 - route['distance']) / 50 * 100)
        volume_score = max(0, (900 - current_volume) / 900 * 100)
        time_score = max(0, (120 - route['estimated_time']) / 120 * 100)
        
        if route['type'] == 'alternative' and current_volume >= 600:
            alternative_bonus = 20
        else:
            alternative_bonus = 0
        
        total_score = (
            prediction_score * 0.3 +
            volume_score * 0.25 +
            time_score * 0.25 +
            distance_score * 0.2 +
            alternative_bonus
        )
        
        route['ai_score'] = total_score
        route['prediction_score'] = prediction_score
        route['confidence'] = min(95, max(70, total_score))
        
        if total_score > best_score:
            best_score = total_score
            best_route = route
    
    return best_route, routes

def create_traffic_map_with_routes(origin, destination, routes, locations, best_route=None):
    """Buat peta dengan rute dan monitoring lalu lintas"""
    bengkulu_center = [-3.8004, 102.2655]
    m = folium.Map(location=bengkulu_center, zoom_start=11)
    
    # Ambil monitoring points dari session state atau generate jika belum ada
    if 'monitoring_points' not in st.session_state:
        st.session_state.monitoring_points, _ = generate_stable_traffic_data()
    
    monitoring_points = st.session_state.monitoring_points
    
    for point in monitoring_points:
        level, icon = classify_traffic_level(point["volume"])
        color = {"🔴": "red", "🟠": "orange", "🟡": "yellow", "🟢": "green", "🔵": "blue"}[icon]
        
        folium.CircleMarker(
            location=[point["lat"], point["lng"]],
            radius=10,
            popup=f"<b>{point['name']}</b><br>Volume: {point['volume']} kendaraan/jam<br>Status: {level}<br>🤖 AI Monitoring Aktif",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
    
    # Tambahkan marker untuk origin dan destination
    origin_coord = locations[origin]
    dest_coord = locations[destination]
    
    folium.Marker(
        location=[origin_coord['lat'], origin_coord['lng']],
        popup=f"🚀 Titik Awal: {origin}",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        location=[dest_coord['lat'], dest_coord['lng']],
        popup=f"🎯 Tujuan: {destination}",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    # Tambahkan rute
    route_colors = ['blue', 'purple', 'orange', 'darkgreen']
    for i, route in enumerate(routes):
        color = route_colors[i % len(route_colors)]
        
        route_coords = [[wp['lat'], wp['lng']] for wp in route['waypoints']]
        
        if best_route and route['name'] == best_route['name']:
            weight = 6
            opacity = 1.0
            color = 'darkgreen'
        elif route['type'] == 'main':
            weight = 5
            opacity = 0.8
        else:
            weight = 3
            opacity = 0.6
        
        if route['type'] == 'alternative':
            dash_array = '10, 5'
        else:
            dash_array = None
        
        folium.PolyLine(
            locations=route_coords,
            color=color,
            weight=weight,
            opacity=opacity,
            dash_array=dash_array
).add_to(m)
        
        # Tambahkan popup untuk setiap rute
        route_info = f"""
        <b>{route['name']}</b><br>
        📍 {' → '.join(route['waypoint_names'])}<br>
        📏 Jarak: {route['distance']:.2f} km<br>
        🚗 Volume: {route['traffic_volume']} kendaraan/jam<br>
        ⏱️ Estimasi: {route['estimated_time']:.0f} menit<br>
        🚦 Status: {route['congestion_level']}<br>
        🤖 AI Score: {route.get('ai_score', 0):.1f}/100
        """
        
        # Tambahkan marker di titik tengah rute untuk info
        if len(route_coords) > 1:
            mid_point = route_coords[len(route_coords)//2]
            folium.Marker(
                location=mid_point,
                popup=route_info,
                icon=folium.Icon(color='lightblue', icon='info-sign')
            ).add_to(m)
    
    return m

def generate_stable_traffic_data(seed=42):
    """Generate data lalu lintas yang stabil untuk monitoring"""
    np.random.seed(seed)
    locations = get_bengkulu_locations()
    current_time = datetime.now()
    
    # Buat titik monitoring
    monitoring_points = []
    location_names = list(locations.keys())
    
    for i, (name, coord) in enumerate(locations.items()):
        # Simulasi volume berdasarkan waktu dan lokasi
        hour = current_time.hour
        
        # Pola lalu lintas berdasarkan jenis lokasi
        if "Terminal" in name or "Bandara" in name or "Pelabuhan" in name:
            base_volume = 600
        elif "Jl." in name or "Raya" in name:
            base_volume = 500
        elif "Mall" in name or "Pasar" in name:
            base_volume = 400
        elif "Universitas" in name or "Kampus" in name:
            base_volume = 300
        else:
            base_volume = 250
        
        # Faktor waktu
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            time_factor = 1.8
        elif 12 <= hour <= 14:
            time_factor = 1.3
        elif 22 <= hour or hour <= 5:
            time_factor = 0.4
        else:
            time_factor = 1.0
        
        # Noise
        noise = np.random.randint(-100, 100)
        
        volume = max(50, min(900, int(base_volume * time_factor + noise)))
        
        monitoring_points.append({
            "name": name,
            "lat": coord["lat"],
            "lng": coord["lng"],
            "volume": volume,
            "timestamp": current_time
        })
    
    return monitoring_points, current_time

def create_traffic_dashboard():
    """Buat dashboard monitoring lalu lintas"""
    st.title("🚦 Smart Traffic Bengkulu AI")
    st.markdown("### 🤖 Sistem Monitoring Lalu Lintas Berbasis Artificial Intelligence")
    
    # Sidebar untuk kontrol
    with st.sidebar:
        st.header("🎛️ Kontrol AI")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=True)
        
        if auto_refresh:
            refresh_interval = st.slider("Interval refresh (detik)", 5, 60, 30)
            
        # Train model button
        if st.button("🧠 Train AI Model", type="primary"):
            predictor = get_lstm_predictor()
            with st.spinner("🤖 Training AI Model..."):
                success, message = predictor.train_model()
                if success:
                    st.session_state.model_trained = True
                    st.session_state.training_message = message
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
        
        # Status model
        if st.session_state.model_trained:
            st.success("🧠 AI Model: Siap")
            st.info(st.session_state.training_message)
        else:
            st.warning("🧠 AI Model: Belum dilatih")
        
        st.markdown("---")
        st.markdown("### 📊 Fitur AI")
        st.markdown("""
        - 🔮 Prediksi lalu lintas real-time
        - 🛣️ Rekomendasi rute optimal
        - 🚦 Analisis kemacetan cerdas
        - 📱 Monitoring 24/7
        """)
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Generate atau ambil data monitoring
    if 'monitoring_points' not in st.session_state or 'last_update' not in st.session_state:
        st.session_state.monitoring_points, st.session_state.last_update = generate_stable_traffic_data()
    
    monitoring_points = st.session_state.monitoring_points
    current_time = st.session_state.last_update
    
    # Hitung statistik
    volumes = [point['volume'] for point in monitoring_points]
    avg_volume = np.mean(volumes)
    max_volume = max(volumes)
    min_volume = min(volumes)
    
    # Klasifikasi tingkat kemacetan kota
    city_level, city_icon = classify_traffic_level(avg_volume)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Volume Rata-rata</h3>
            <h2>{avg_volume:.0f}</h2>
            <p>kendaraan/jam</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔥 Puncak Kemacetan</h3>
            <h2>{max_volume}</h2>
            <p>kendaraan/jam</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🟢 Minimum</h3>
            <h2>{min_volume}</h2>
            <p>kendaraan/jam</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="ai-card">
            <h3>🤖 Status AI</h3>
            <h2>{city_icon} {city_level}</h2>
            <p>Confidence: 95%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Alert sistem
    if avg_volume >= 700:
        st.markdown(f"""
        <div class="alert-high">
            <h3>🚨 PERINGATAN TINGGI - AI ALERT</h3>
            <p>Sistem AI mendeteksi kemacetan parah di seluruh kota. Volume rata-rata: {avg_volume:.0f} kendaraan/jam</p>
            <p>🤖 Rekomendasi AI: Hindari perjalanan non-esensial dan gunakan rute alternatif</p>
        </div>
        """, unsafe_allow_html=True)
    elif avg_volume >= 500:
        st.markdown(f"""
        <div class="alert-medium">
            <h3>⚠️ PERINGATAN SEDANG - AI MONITORING</h3>
            <p>Sistem AI memantau peningkatan lalu lintas. Volume rata-rata: {avg_volume:.0f} kendaraan/jam</p>
            <p>🤖 Saran AI: Rencanakan perjalanan dengan waktu tambahan</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-low">
            <h3>✅ KONDISI NORMAL - AI CONFIRMED</h3>
            <p>Sistem AI mengkonfirmasi kondisi lalu lintas normal. Volume rata-rata: {avg_volume:.0f} kendaraan/jam</p>
            <p>🤖 AI Status: Semua sistem optimal</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs untuk berbagai fitur
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Peta Real-time", "🛣️ Route Planner", "📈 Analisis AI", "🔮 Prediksi"])
    
    with tab1:
        st.subheader("🗺️ Peta Monitoring Real-time")
        
        # Buat peta sederhana untuk monitoring
        bengkulu_center = [-3.8004, 102.2655]
        m = folium.Map(location=bengkulu_center, zoom_start=11)
        
        # Tambahkan monitoring points
        for point in monitoring_points:
            level, icon = classify_traffic_level(point["volume"])
            color = {"🔴": "red", "🟠": "orange", "🟡": "yellow", "🟢": "green", "🔵": "blue"}[icon]
            
            folium.CircleMarker(
                location=[point["lat"], point["lng"]],
                radius=8,
                popup=f"<b>{point['name']}</b><br>Volume: {point['volume']} kendaraan/jam<br>Status: {level}<br>🤖 AI Monitoring Aktif",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        # Tampilkan peta
        st_folium(m, width=700, height=500)
        
        # Info update
        st.info(f"🕒 Terakhir update: {current_time.strftime('%H:%M:%S')} - 🤖 AI Monitoring Aktif")
    
    with tab2:
        st.subheader("🛣️ AI Route Planner")
        
        locations = get_bengkulu_locations()
        location_names = list(locations.keys())
        
        col1, col2 = st.columns(2)
        
        with col1:
            origin = st.selectbox("📍 Titik Awal", location_names, index=0)
        
        with col2:
            destination = st.selectbox("🎯 Tujuan", location_names, index=1)
        
        if st.button("🚀 Cari Rute Optimal", type="primary"):
            if origin != destination:
                with st.spinner("🤖 AI sedang menganalisis rute optimal..."):
                    routes = get_stable_route_data(origin, destination, locations)
                    predictor = get_lstm_predictor()
                    
                    # Dapatkan rekomendasi AI
                    best_route, analyzed_routes = get_route_recommendation(routes, predictor)
                    
                    # Simpan di session state
                    st.session_state.current_routes = analyzed_routes
                    st.session_state.best_route = best_route
                    st.session_state.route_origin = origin
                    st.session_state.route_destination = destination
        
        # Tampilkan hasil jika ada
        if hasattr(st.session_state, 'current_routes') and st.session_state.current_routes:
            st.markdown("### 🤖 Hasil Analisis AI")
            
            # Tampilkan rute terbaik
            if st.session_state.best_route:
                best = st.session_state.best_route
                st.markdown(f"""
                <div class="best-route">
                    <h3>🏆 REKOMENDASI AI TERBAIK</h3>
                    <h4>{best['name']}</h4>
                    <p><strong>Rute:</strong> {' → '.join(best['waypoint_names'])}</p>
                    <p><strong>Jarak:</strong> {best['distance']:.2f} km</p>
                    <p><strong>Estimasi Waktu:</strong> {best['estimated_time']:.0f} menit</p>
                    <p><strong>Status Lalu Lintas:</strong> {best['congestion_icon']} {best['congestion_level']}</p>
                    <p><strong>🤖 AI Confidence:</strong> {best['confidence']:.1f}%</p>
                    <p><strong>🔬 AI Score:</strong> {best['ai_score']:.1f}/100</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Tampilkan semua rute
            st.markdown("### 📋 Semua Opsi Rute")
            
            for i, route in enumerate(st.session_state.current_routes):
                if route['type'] == 'main':
                    card_class = "route-card"
                else:
                    card_class = "alternative-route"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <h4>{route['name']}</h4>
                    <p><strong>Via:</strong> {' → '.join(route['waypoint_names'])}</p>
                    <p><strong>Jarak:</strong> {route['distance']:.2f} km | <strong>Waktu:</strong> {route['estimated_time']:.0f} menit</p>
                    <p><strong>Status:</strong> {route['congestion_icon']} {route['congestion_level']} ({route['traffic_volume']} kendaraan/jam)</p>
                    <p><strong>🤖 AI Score:</strong> {route['ai_score']:.1f}/100 | <strong>Confidence:</strong> {route['confidence']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Tampilkan peta dengan rute
            st.markdown("### 🗺️ Visualisasi Rute")
            route_map = create_traffic_map_with_routes(
                st.session_state.route_origin,
                st.session_state.route_destination,
                st.session_state.current_routes,
                locations,
                st.session_state.best_route
            )
            st_folium(route_map, width=700, height=500)
    
    with tab3:
        st.subheader("📈 Analisis Lalu Lintas AI")
        
        # Analisis distribusi kemacetan
        congestion_counts = {}
        for point in monitoring_points:
            level, _ = classify_traffic_level(point['volume'])
            congestion_counts[level] = congestion_counts.get(level, 0) + 1
        
        # Chart distribusi
        if congestion_counts:
            labels = list(congestion_counts.keys())
            values = list(congestion_counts.values())
            
            fig = go.Figure(data=[
                go.Bar(x=labels, y=values, 
                       marker_color=['red' if 'Padat' in label else 'orange' if 'Sedang' in label else 'green' 
                                   for label in labels])
            ])
            fig.update_layout(
                title="🤖 Distribusi Tingkat Kemacetan (AI Analysis)",
                xaxis_title="Tingkat Kemacetan",
                yaxis_title="Jumlah Lokasi",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top 5 lokasi terpadat
        st.markdown("### 🔥 Top 5 Lokasi Terpadat")
        sorted_points = sorted(monitoring_points, key=lambda x: x['volume'], reverse=True)
        
        for i, point in enumerate(sorted_points[:5]):
            level, icon = classify_traffic_level(point['volume'])
            recommendations = get_traffic_recommendations(level, point['volume'])
            
            st.markdown(f"""
            <div class="ai-info">
                <h4>#{i+1} {point['name']}</h4>
                <p><strong>Volume:</strong> {point['volume']} kendaraan/jam</p>
                <p><strong>Status:</strong> {icon} {level}</p>
                <p><strong>🤖 AI Recommendations:</strong></p>
                <ul>
                    {''.join([f'<li>{rec}</li>' for rec in recommendations[:2]])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("🔮 Prediksi Lalu Lintas AI")
        
        predictor = get_lstm_predictor()
        
        if not predictor.is_trained:
            st.warning("⚠️ Model AI belum dilatih. Klik 'Train AI Model' di sidebar untuk mengaktifkan prediksi.")
        else:
            st.success("✅ Model AI siap untuk prediksi!")
            
            # Pilih lokasi untuk prediksi
            selected_location = st.selectbox("📍 Pilih lokasi untuk prediksi", 
                                           [point['name'] for point in monitoring_points])
            
            if st.button("🔮 Prediksi 3 Jam Kedepan", type="primary"):
                # Cari data lokasi yang dipilih
                selected_point = next((p for p in monitoring_points if p['name'] == selected_location), None)
                
                if selected_point:
                    current_volume = selected_point['volume']
                    
                    # Generate historical data (simulasi)
                    historical_data = [current_volume + np.random.randint(-50, 50) for _ in range(12)]
                    
                    # Prediksi
                    predictions, status = predictor.predict_next_hours(historical_data, 3)
                    
                    if predictions:
                        st.markdown("### 🔮 Hasil Prediksi AI")
                        
                        # Buat data untuk chart
                        hours = [datetime.now() + timedelta(hours=i) for i in range(1, 4)]
                        hour_labels = [h.strftime("%H:%M") for h in hours]
                        
                        # Chart prediksi
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=hour_labels,
                            y=predictions,
                            mode='lines+markers',
                            name='Prediksi AI',
                            line=dict(color='cyan', width=3),
                            marker=dict(size=10)
                        ))
                        
                        fig.add_hline(y=current_volume, line_dash="dash", 
                                    annotation_text=f"Volume Saat Ini: {current_volume}")
                        
                        fig.update_layout(
                            title=f"🔮 Prediksi Lalu Lintas AI - {selected_location}",
                            xaxis_title="Waktu",
                            yaxis_title="Volume Kendaraan/Jam",
                            template="plotly_dark"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Prediksi detail
                        for i, (hour, pred_volume) in enumerate(zip(hours, predictions)):
                            level, icon = classify_traffic_level(pred_volume)
                            recommendations = get_traffic_recommendations(level, pred_volume, ai_confidence=predictor.confidence if hasattr(predictor, 'confidence') else 85)

                            
                            st.markdown(f"""
                            <div class="ai-info">
                                <h4>🕐 {hour.strftime("%H:%M")} - Prediksi #{i+1}</h4>
                                <p><strong>Volume Prediksi:</strong> {pred_volume:.0f} kendaraan/jam</p>
                                <p><strong>Status Prediksi:</strong> {icon} {level}</p>
                                <p><strong>🤖 AI Recommendations:</strong></p>
                                <ul>
                                    {''.join([f'<li>{rec}</li>' for rec in recommendations[:3]])}
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.error(f"❌ Gagal melakukan prediksi: {status}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h3>🤖 Smart Traffic Bengkulu AI</h3>
        <p>Powered by Artificial Intelligence & Machine Learning</p>
        <p>🧠 LSTM Neural Network | 📊 Real-time Analytics | 🔮 Predictive Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

# Jalankan aplikasi
if __name__ == "__main__":
    create_traffic_dashboard()