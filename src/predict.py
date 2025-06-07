import numpy as np
from tensorflow.keras.models import load_model

def predict_next(model_path, recent_data):
    model = load_model(model_path)
    prediction = model.predict(recent_data)
    return prediction
