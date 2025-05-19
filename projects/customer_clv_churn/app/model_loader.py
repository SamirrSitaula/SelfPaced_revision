import joblib
import os

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../outputs/models'))

model = joblib.load(os.path.join(MODEL_DIR, 'logistic_model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
