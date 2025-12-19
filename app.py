from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import sys
import traceback
import json
from pathlib import Path

print("=" * 50, flush=True)
print(" STARTING FLASK APPLICATION", flush=True)
print("=" * 50, flush=True)

# ==== TENTUKAN PATHS DENGAN BENAR ====
BASE_DIR = Path(__file__).resolve().parent
print(f"Base directory: {BASE_DIR}", flush=True)
print(f"Current working directory: {os.getcwd()}", flush=True)

# Cari file .pkl di berbagai lokasi
possible_paths = [
    BASE_DIR / 'model.pkl',
    BASE_DIR.parent / 'model.pkl',
    Path('model.pkl'),
    Path('/home/cobh4463/aplikasi-rekomendasi-jurusan/model.pkl'),  # Sesuaikan dengan path hosting Anda
]

print(f"Files in current directory: {os.listdir('.')}", flush=True)
if os.path.exists(BASE_DIR):
    print(f"Files in BASE_DIR: {os.listdir(BASE_DIR)}", flush=True)

# ==== CEK FILE .PKL ADA ATAU TIDAK ====
print("Checking if model files exist...", flush=True)
required_files = ['model.pkl', 'scaler.pkl', 'label_encoder.pkl']

model_path = None
scaler_path = None
encoder_path = None

# Cari model.pkl
for path in possible_paths:
    if path.exists():
        model_path = str(path)
        print(f"Found model.pkl at: {model_path}", flush=True)
        break

if not model_path:
    print("WARNING: model.pkl not found in standard locations", flush=True)
    model_path = 'model.pkl'

for path in [BASE_DIR / 'scaler.pkl', Path('scaler.pkl'), BASE_DIR.parent / 'scaler.pkl']:
    if path.exists():
        scaler_path = str(path)
        break
if not scaler_path:
    scaler_path = 'scaler.pkl'

for path in [BASE_DIR / 'label_encoder.pkl', Path('label_encoder.pkl'), BASE_DIR.parent / 'label_encoder.pkl']:
    if path.exists():
        encoder_path = str(path)
        break
if not encoder_path:
    encoder_path = 'label_encoder.pkl'

print(f"Will attempt to load from: {model_path}, {scaler_path}, {encoder_path}", flush=True)

app = Flask(__name__)
CORS(app)

# ==== Load Model dengan Error Handling ====
model = None
scaler = None
label_encoder = None

try:
    print(f"Attempting to load {model_path}...", flush=True)
    model = joblib.load(model_path)
    size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"model.pkl loaded successfully! ({size:.2f} MB)", flush=True)
except Exception as e:
    print(f"ERROR loading model.pkl: {str(e)}", flush=True)
    traceback.print_exc()
    print("WARNING: Continuing without model (API will return errors)", flush=True)
    model = None

try:
    print(f"Attempting to load {scaler_path}...", flush=True)
    scaler = joblib.load(scaler_path)
    size = os.path.getsize(scaler_path) / (1024 * 1024)
    print(f"scaler.pkl loaded successfully! ({size:.2f} MB)", flush=True)
except Exception as e:
    print(f"ERROR loading scaler.pkl: {str(e)}", flush=True)
    traceback.print_exc()
    print("WARNING: Continuing without scaler (API will return errors)", flush=True)
    scaler = None

try:
    print(f"Attempting to load {encoder_path}...", flush=True)
    label_encoder = joblib.load(encoder_path)
    size = os.path.getsize(encoder_path) / (1024 * 1024)
    print(f"label_encoder.pkl loaded successfully! ({size:.2f} MB)", flush=True)
except Exception as e:
    print(f"ERROR loading label_encoder.pkl: {str(e)}", flush=True)
    traceback.print_exc()
    print("WARNING: Continuing without encoder (API will return errors)", flush=True)
    label_encoder = None

models_ready = all([model, scaler, label_encoder])
print(f"Models ready: {models_ready}", flush=True)
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
    return jsonify({
        'message': 'API Rekomendasi Jurusan - Flask is running!',
        'models_loaded': models_ready,
        'endpoints': {
            'health': '/health',
            'predict': '/predict'
        }
    }), 200

@app.route('/health')
def health():
    """Health check endpoint"""
    print("Health check accessed", flush=True)
    return jsonify({
        'status': 'healthy' if models_ready else 'degraded',
        'message': 'API is running',
        'models_loaded': models_ready
    }), 200 if models_ready else 503

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Predict endpoint accessed", flush=True)
        
        # Cek apakah models sudah loaded
        if not models_ready:
            print("ERROR: Models not loaded", flush=True)
            return jsonify({
                'error': 'Model tidak tersedia. Hubungi administrator.',
                'jurusan': None,
                'deskripsi': None
            }), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'JSON data tidak diterima',
                'jurusan': None,
                'deskripsi': None
            }), 400

        # Cek apakah fitur lengkap
        missing_features = [f for f in FEATURE_COLUMNS if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Input tidak lengkap. Missing: {missing_features}',
                'jurusan': None,
                'deskripsi': None
            }), 400

        # Validasi tipe data
        try:
            input_values = []
            for f in FEATURE_COLUMNS:
                val = float(data[f])
                input_values.append(val)
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': f'Nilai harus berupa angka. Error: {str(e)}',
                'jurusan': None,
                'deskripsi': None
            }), 400

        input_array = np.array(input_values).reshape(1, -1)
        print(f"Input shape: {input_array.shape}", flush=True)

        # ==== Preprocessing: Scaling ====
        input_scaled = scaler.transform(input_array)

        # ==== Prediksi ====
        hasil_encoded = model.predict(input_scaled)
        print(f"Prediction encoded: {hasil_encoded}", flush=True)

        # ==== Decode ke nama jurusan ====
        hasil_jurusan = label_encoder.inverse_transform(hasil_encoded)[0]
        print(f"Prediction decoded: {hasil_jurusan}", flush=True)

        # ==== Deskripsi Map ====
        deskripsi_map = {
            'Teknik': 'Jurusan Teknik cocok untuk Anda yang memiliki kemampuan kuat di Matematika, Fisika, dan Kimia. Jurusan ini membuka peluang karir di bidang teknik sipil, teknik mesin, teknik elektro, teknik informatika, dan berbagai bidang rekayasa teknologi.',
            
            'Kesehatan': 'Jurusan Kesehatan cocok untuk Anda yang memiliki nilai baik di Biologi, Kimia, dan Fisika. Jurusan ini membuka peluang karir di bidang kedokteran, keperawatan, farmasi, kesehatan masyarakat, gizi, dan profesi kesehatan lainnya.',
            
            'Ilmu Komunikasi': 'Jurusan Ilmu Komunikasi cocok untuk Anda yang memiliki kemampuan komunikasi yang baik dalam Bahasa Indonesia dan Bahasa Inggris. Jurusan ini membuka peluang karir di bidang jurnalisme, public relations, broadcasting, periklanan, dan media digital.',
            
            'Matematika Murni': 'Jurusan Matematika Murni cocok untuk Anda yang memiliki kemampuan analitis dan logika kuat dalam Matematika. Jurusan ini membuka peluang karir di bidang penelitian matematika, aktuaria, data science, statistika, dan pengembangan algoritma.',
            
            'Pertanian dan Kehutanan': 'Jurusan Pertanian dan Kehutanan cocok untuk Anda yang memiliki minat di Biologi, Kimia, dan kepedulian terhadap lingkungan. Jurusan ini membuka peluang karir di bidang agronomi, peternakan, kehutanan, konservasi alam, dan teknologi pertanian.',
            
            'Fisika Murni': 'Jurusan Fisika Murni cocok untuk Anda yang memiliki kemampuan kuat di Fisika dan Matematika. Jurusan ini membuka peluang karir di bidang penelitian fisika, instrumentasi, energi, astronomi, dan teknologi nuklir.',
            
            'Sastra Inggris': 'Jurusan Sastra Inggris cocok untuk Anda yang memiliki kemampuan menonjol dalam Bahasa Inggris dan apresiasi terhadap karya sastra. Jurusan ini membuka peluang karir di bidang penerjemahan, kritik sastra, penerbitan, content writing, dan pengajaran bahasa.',
            
            'Pendidikan Bahasa Inggris': 'Jurusan Pendidikan Bahasa Inggris cocok untuk Anda yang memiliki kemampuan Bahasa Inggris yang baik dan minat dalam dunia pendidikan. Jurusan ini membuka peluang karir sebagai guru bahasa Inggris, pengembang kurikulum, instruktur pelatihan, dan pendidik profesional.',
            
            'Akuntansi': 'Jurusan Akuntansi cocok untuk Anda yang memiliki kemampuan Matematika dan Ekonomi yang baik serta teliti dalam angka. Jurusan ini membuka peluang karir di bidang akuntan publik, auditor, konsultan pajak, analis keuangan, dan manajer akuntansi.',
            
            'Agribisnis': 'Jurusan Agribisnis cocok untuk Anda yang memiliki nilai baik di Ekonomi, Biologi, dan Matematika. Jurusan ini membuka peluang karir di bidang manajemen agribisnis, kewirausahaan pertanian, pemasaran produk pertanian, dan pengembangan bisnis agro-industri.',
            
            'Sosiologi': 'Jurusan Sosiologi cocok untuk Anda yang memiliki kemampuan memahami dinamika sosial melalui mata pelajaran Sosiologi, PPKN, dan Sejarah. Jurusan ini membuka peluang karir di bidang penelitian sosial, pengembangan masyarakat, kebijakan publik, dan konsultan sosial.',
            
            'PGSD': 'Jurusan PGSD (Pendidikan Guru Sekolah Dasar) cocok untuk Anda yang memiliki nilai seimbang di berbagai mata pelajaran dan minat dalam pendidikan anak. Jurusan ini membuka peluang karir sebagai guru SD, pengembang pendidikan dasar, dan tenaga kependidikan profesional.',
            
            'Hukum': 'Jurusan Hukum cocok untuk Anda yang memiliki kemampuan baik di PPKN, Bahasa Indonesia, dan Sejarah dengan kemampuan analisis yang kuat. Jurusan ini membuka peluang karir di bidang advokat, hakim, jaksa, notaris, konsultan hukum, dan legal officer.',
            
            'Psikologi': 'Jurusan Psikologi cocok untuk Anda yang memiliki kemampuan memahami perilaku manusia melalui Sosiologi, Biologi, dan PPKN. Jurusan ini membuka peluang karir di bidang psikolog klinis, konselor, HRD, peneliti perilaku, dan psikolog industri.',
            
            'Manajemen Bisnis': 'Jurusan Manajemen Bisnis cocok untuk Anda yang memiliki kemampuan di Ekonomi, Matematika, dan Bahasa Inggris. Jurusan ini membuka peluang karir di bidang manajer bisnis, entrepreneur, konsultan manajemen, business analyst, dan marketing manager.',
            
            'Seni Rupa/DKV': 'Jurusan Seni Rupa/DKV (Desain Komunikasi Visual) cocok untuk Anda yang memiliki kemampuan Seni Budaya yang menonjol dan kreativitas tinggi. Jurusan ini membuka peluang karir di bidang desain grafis, ilustrator, animator, creative director, dan seniman visual.',
            
            'Ekonomi': 'Jurusan Ekonomi cocok untuk Anda yang memiliki kemampuan kuat di Ekonomi dan Matematika dengan pemahaman tentang isu sosial. Jurusan ini membuka peluang karir di bidang ekonom, analis ekonomi, perbankan, konsultan ekonomi, dan peneliti kebijakan ekonomi.',
            
            'Arsitektur': 'Jurusan Arsitektur cocok untuk Anda yang memiliki kemampuan di Matematika, Fisika, dan Seni Budaya dengan kreativitas dalam desain. Jurusan ini membuka peluang karir di bidang arsitek, perencana kota, desainer interior, konsultan bangunan, dan landscape architect.',
            
            'Sejarah': 'Jurusan Sejarah cocok untuk Anda yang memiliki kemampuan baik di Sejarah, Bahasa Indonesia, dan PPKN dengan minat pada masa lalu. Jurusan ini membuka peluang karir di bidang sejarawan, peneliti, arkeolog, kurator museum, guru sejarah, dan penulis sejarah.',
            
            'Biologi Murni': 'Jurusan Biologi Murni cocok untuk Anda yang memiliki kemampuan kuat di Biologi dan Kimia dengan minat pada sains kehidupan. Jurusan ini membuka peluang karir di bidang peneliti biologi, bioteknologi, konservasi, mikrobiologi, dan pengembangan farmasi.'
        }
        
        deskripsi = deskripsi_map.get(
            hasil_jurusan, 
            f'Jurusan {hasil_jurusan} direkomendasikan berdasarkan analisis nilai mata pelajaran Anda.'
        )

        print(f"Prediction successful: {hasil_jurusan}", flush=True)

        return jsonify({
            'jurusan': hasil_jurusan,
            'deskripsi': deskripsi,
            'status': 'success'
        }), 200

    except Exception as e:
        print(f"Error in predict: {str(e)}", flush=True)
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'jurusan': None,
            'deskripsi': None,
            'status': 'error'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)