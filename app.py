from flask import Flask, request, jsonify, render_template
from main_qa_module import PDFProcessor, QAEngine
from dotenv import load_dotenv
import os
import time
import traceback
from threading import Lock

# ====== Load Environment Variables ======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY tidak ditemukan di file .env")

# ====== Inisialisasi Flask App ======
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# ====== Inisialisasi Sistem QA ======
def initialize_qa_system():
    try:
        start_time = time.time()
        processor = PDFProcessor()
        pdf_path = 'dokumen/dataset_pompav.pdf'
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"❌ File PDF tidak ditemukan: {pdf_path}")
        
        print("🛠️ Memulai inisialisasi sistem QA...")

        # Initialize vector store (load or create new)
        if not processor.initialize_vector_store(pdf_path):
            raise ValueError("❌ Gagal memuat atau membuat vektor store")

        qa_engine = QAEngine(processor.vector_store)
        
        print(f"✅ Sistem QA siap digunakan dalam {time.time()-start_time:.2f} detik")
        return qa_engine
        
    except Exception as e:
        print(f"❌ Gagal inisialisasi sistem QA: {str(e)}")
        traceback.print_exc()
        return None

# Initialize QA Engine globally
qa_engine = initialize_qa_system()

# Lock untuk thread-safe processing
processing_lock = Lock()

# ====== ROUTES ======
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    # Health check route to check the status of the QA system
    return jsonify({
        'status': 'ready' if qa_engine else 'error',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/ask', methods=['POST'])
def ask():
    if not qa_engine:
        return jsonify({
            'response': '⚠️ Sistem QA belum siap. Silakan coba lagi nanti.',
            'status': 'error'
        }), 503

    # Cek jika sistem sedang sibuk
    if not processing_lock.acquire(blocking=False):
        return jsonify({
            'response': '⚠️ Sistem sedang memproses pertanyaan sebelumnya. Silakan tunggu...',
            'status': 'busy'
        }), 429

    try:
        # Parse JSON input
        data = request.get_json()
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({'response': '⚠️ Pertanyaan kosong tidak dapat diproses'}), 400

        print(f"\n📩 Pertanyaan diterima: {user_message}")
        start_time = time.time()
        response = qa_engine.generate_response(user_message)
        processing_time = time.time() - start_time

        result = {
            'status': 'success',
            'processing_time': f"{processing_time:.2f} detik"
        }

        if isinstance(response, tuple) and len(response) >= 2:
            answer, scores = response
            result.update({
                'response': answer,
                'f1_score': round(scores[2], 2) if scores and len(scores) > 2 else None
            })
            
            # Print to terminal
            print(f"\n📨 Jawaban: {answer}")
            if scores[2] > 0:
                print(f"⏱ Waktu pemrosesan: {processing_time:.2f} detik")
                print(f"📊 Skor F1: {scores[2]:.4f}")
        else:
            result['response'] = response
            print(f"\n📨 Jawaban: {response}")

        return jsonify(result)
            
    except Exception as e:
        error_msg = f'❌ Terjadi kesalahan: {str(e)}'
        print(error_msg)
        return jsonify({
            'response': error_msg,
            'status': 'error'
        }), 500
    finally:
        processing_lock.release()

# ====== MAIN ======
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
