import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['volume'] = df['volume'].fillna(method='ffill')  # isi nilai kosong
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['volume']])
    return scaled, scaler
