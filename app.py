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

        # ==== UBAH RETURN INI ====
        return jsonify({
            'jurusan': hasil_jurusan,
            'deskripsi': deskripsi
        })

    except Exception as e:
        print(f"Error in predict: {str(e)}", flush=True)
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'jurusan': None,
            'deskripsi': 'Terjadi kesalahan saat melakukan prediksi'
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port)