from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import sys
import traceback

print("=" * 50, flush=True)
print(" STARTING FLASK APPLICATION", flush=True)
print("=" * 50, flush=True)

# ==== CEK FILE .PKL ADA ATAU TIDAK ====
print("Checking if model files exist...", flush=True)
print(f"Current working directory: {os.getcwd()}", flush=True)
print(f"Files in current directory: {os.listdir('.')}", flush=True)

required_files = ['model.pkl', 'scaler.pkl', 'label_encoder.pkl']
for file in required_files:
    if os.path.exists(file):
        size = os.path.getsize(file) / (1024 * 1024)  # MB
        print(f"{file} exists ({size:.2f} MB)", flush=True)
    else:
        print(f"{file} NOT FOUND!", flush=True)

app = Flask(__name__)
CORS(app)

# ==== Load Model dengan Error Handling ====
model = None
scaler = None
label_encoder = None

try:
    print("Attempting to load model.pkl...", flush=True)
    model = joblib.load('model.pkl')
    print("model.pkl loaded successfully!", flush=True)
except Exception as e:
    print(f"ERROR loading model.pkl: {str(e)}", flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    print("Attempting to load scaler.pkl...", flush=True)
    scaler = joblib.load('scaler.pkl')
    print("scaler.pkl loaded successfully!", flush=True)
except Exception as e:
    print(f"ERROR loading scaler.pkl: {str(e)}", flush=True)
    traceback.print_exc()
    sys.exit(1)

try:
    print("Attempting to load label_encoder.pkl...", flush=True)
    label_encoder = joblib.load('label_encoder.pkl')
    print("label_encoder.pkl loaded successfully!", flush=True)
except Exception as e:
    print(f"ERROR loading label_encoder.pkl: {str(e)}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("All models loaded successfully!", flush=True)
print("=" * 50, flush=True)

# ==== Kolom Fitur ====
FEATURE_COLUMNS = [
    "Matematika",
    "Fisika",
    "Kimia",
    "Biologi",
    "Ekonomi",
    "Sosiologi",
    "Agama Islam",
    "PPKN",
    "Sejarah",
    "Seni Budaya",
    "Penjas",
    "B_Indonesia",
    "B_Inggris"
]

@app.route('/')
def home():
    print("Home endpoint accessed", flush=True)
    return "API Rekomendasi Jurusan - Flask is running!"

@app.route('/health')
def health():
    """Health check endpoint"""
    print("Health check accessed", flush=True)
    return jsonify({
        'status': 'healthy',
        'message': 'API is running',
        'models_loaded': True
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Predict endpoint accessed", flush=True)
        data = request.get_json()

        # Cek apakah fitur lengkap
        missing_features = [f for f in FEATURE_COLUMNS if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Input tidak lengkap: {missing_features}'
            }), 400

        # Ambil nilai input sesuai urutan kolom
        input_values = [data[f] for f in FEATURE_COLUMNS]
        input_array = np.array(input_values).reshape(1, -1)

        # ==== Preprocessing: Scaling ====
        input_scaled = scaler.transform(input_array)

        # ==== Prediksi ====
        hasil_encoded = model.predict(input_scaled)

        # ==== Decode ke nama jurusan ====
        hasil_jurusan = label_encoder.inverse_transform(hasil_encoded)[0]

        print(f"Prediction successful: {hasil_jurusan}", flush=True)

        return jsonify({
            'rekomendasi_jurusan': hasil_jurusan,
            'status': 'success'
        })

    except Exception as e:
        print(f"Error in predict: {str(e)}", flush=True)
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port)