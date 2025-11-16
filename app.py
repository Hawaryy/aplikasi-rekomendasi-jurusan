from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ==== Load Model, Scaler, dan Label Encoder ====
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# ==== Kolom Fitur (urutan harus sesuai dengan dataset training) ====
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
    return "✅ API Rekomendasi Jurusan - Flask is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
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

        return jsonify({
            'rekomendasi_jurusan': hasil_jurusan,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)