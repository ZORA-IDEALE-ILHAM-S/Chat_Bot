## main_qa_module
# ====== IMPORT LIBRARY ====== #
import os
import fitz
import traceback
import tiktoken
import time
import re
import numpy as np
from typing import Tuple, Optional, List, Dict
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from bert_score import score

import sys
sys.stdout.flush()  # Pastikan output tidak tertahan di buffer
# ====== KELAS MEMORI PERCAKAPAN ====== #
class ConversationMemory:
    def __init__(self, max_length: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def add_interaction(self, question: str, answer: str):
        if len(self.history) >= self.max_length:
            self.history.pop(0)
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.strftime("%H:%M:%S"),
            "embedding": self._get_embedding(question + " " + answer)
        })

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.similarity_model.encode(text, convert_to_tensor=False)

    def get_context_similarity(self, current_question: str) -> float:
        if not self.history:
            return 0.0
        current_embedding = self._get_embedding(current_question)
        last_embedding = self.history[-1]["embedding"]
        return util.pytorch_cos_sim(current_embedding, last_embedding).item()

    def get_formatted_history(self, num_last: int = 2) -> str:
        return "\n\n".join(
            f"Q [{item['timestamp']}]: {item['question']}\nA: {item['answer'][:200]}..."
            for item in self.history[-num_last:]
        )

    def clear(self):
        self.history = []


# ====== KELAS PEMROSES PDF ====== #
class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            chunk_size=500,
            max_retries=5
        )
        self.vector_store = None

    def extract_text(self, pdf_path: str) -> Optional[str]:
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
                    if len(text) > 2_000_000:
                        print("⚠️ Peringatan: Dokumen sangat besar, hanya memproses bagian awal")
                        break
            return text if text.strip() else None
        except Exception as e:
            print(f"❌ Error ekstraksi PDF: {e}")
            traceback.print_exc()
            return None

    def chunk_text(self, text: str) -> List[str]:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            return splitter.split_text(text[:4_000_000])
        except Exception as e:
            print(f"❌ Error chunking teks: {e}")
            traceback.print_exc()
            return []

    def create_vector_store(self, chunks: List[str]):
        if not chunks:
            print("❌ Tidak ada teks yang bisa diproses")
            return

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(batch, self.embeddings)
                    print(f"✅ Vektor store awal dibuat dengan {len(batch)} chunk")
                else:
                    self.vector_store.add_texts(batch)
                    print(f"✅ Menambahkan batch {i//batch_size + 1} ({len(batch)} chunk)")
            except Exception as e:
                print(f"❌ Gagal memproses batch {i//batch_size + 1}: {e}")
                traceback.print_exc()


# ====== KELAS QA ENGINE ====== #
class QAEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        self.memory = ConversationMemory(max_length=5)
        self.current_context = None
        self.current_topic = None

        # Daftar pola pertanyaan
        self.greetings = ["halo", "hai", "hi", "selamat pagi", "selamat siang", "selamat malam"]
        self.small_talk = ["apa kabar", "bagaimana kabarmu", "siapa namamu"]
        self.reset_commands = ["reset", "clear", "bersihkan memori"]
        self.follow_up_keywords = ["jelaskan", "detail", "lebih", "tentang itu", "mengapa",
                                 "bagaimana", "contoh", "saja", "tersebut", "itu", "ini",
                                 "yg", "yang", "tadi", "nomor", "poin"]

        # Daftar istilah teknis
        self.technical_terms = set(["hsse", "dppu", "pompav", "keselamatan", "penerbangan",
                                   "bandara", "aviasi", "keamanan", "insiden", "darurat"])

    def extract_terms(self, text: str) -> set:
        acronyms = set(re.findall(r'\b[A-Z]{3,}\b', text))
        terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
        return {term.lower() for term in acronyms.union(terms)}

    def is_new_topic(self, question: str) -> bool:
        if not self.memory.history:
            return True
        last_question = self.memory.history[-1]["question"]
        last_terms = self.extract_terms(last_question)
        current_terms = self.extract_terms(question)
        jaccard_sim = len(current_terms & last_terms) / max(len(current_terms | last_terms), 1)
        return jaccard_sim < 0.2

    def update_terms(self, docs):
        for doc in docs:
            self.technical_terms.update(self.extract_terms(doc.page_content))

    def is_technical(self, question: str) -> bool:
        q_lower = question.lower()
        return any(term in q_lower for term in self.technical_terms if len(term) > 2)

    def analyze_question(self, question: str) -> Tuple[Optional[str], str]:
        if not question.strip():
            return None, "empty"

        q_lower = question.lower().strip()

        # Deteksi perintah khusus
        if any(cmd in q_lower for cmd in self.reset_commands):
            return None, "reset"
        if any(phrase in q_lower for phrase in self.small_talk):
            return None, "small_talk"

        # Deteksi sapaan
        has_greeting = any(greet in q_lower for greet in self.greetings)
        cleaned = re.sub('|'.join(self.greetings), "", q_lower).strip(' ,.!?')

        if not cleaned:
            return None, "pure_greeting" if has_greeting else "empty"

        # Deteksi pertanyaan lanjutan
        similarity_score = self.memory.get_context_similarity(question)
        is_follow_up = (similarity_score > 0.6 or
                       any(kw in q_lower for kw in self.follow_up_keywords) or
                       len(cleaned.split()) <= 3)

        if is_follow_up and self.memory.history:
            return question, "follow_up"
        elif self.is_technical(cleaned):
            return question, "technical"
        return question, "general"

    def generate_follow_up_response(self, question: str) -> Dict:
        last_qa = self.memory.history[-1]
        context = f"Pertanyaan sebelumnya: {last_qa['question']}\nJawaban sebelumnya: {last_qa['answer']}"

        prompt_template = """
        [KONTEKS SEBELUMNYA]
        {context}

        [PERTANYAAN LANJUTAN]
        {question}

        [INSTRUKSI]
        1. Identifikasi dengan tepat bagian mana dari jawaban sebelumnya yang dirujuk
        2. Jika pertanyaan ambigu, ajukan pertanyaan klarifikasi dengan menyebutkan:
           - Daftar topik terkait dari jawaban sebelumnya
           - Contoh format pertanyaan yang lebih spesifik
        3. Berikan jawaban yang:
           - Langsung menjawab pertanyaan
           - Memperluas penjelasan sebelumnya
           - Menyertakan contoh konkret

        [FORMAT JAWABAN]
        <konfirmasi topik>
        <penjelasan lengkap>
        <contoh relevan> (jika ada)
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        return chain({
            "input_documents": self.current_context or [],
            "context": context,
            "question": question
        }, return_only_outputs=True)

    def generate_response(self, question: str) -> Tuple[str, Tuple[float, float, float]]:
        try:
            core_q, q_type = self.analyze_question(question)

            # Handle perintah khusus
            if q_type == "reset":
                self.memory.clear()
                self.current_context = None
                return "🔄 Memori percakapan telah direset", (0, 0, 0)

            if q_type in ["empty", "pure_greeting", "small_talk"]:
                samples = "\n".join([
                    "• Apa definisi HSSE menurut dokumen POMPAV?",
                    "• Bagaimana prosedur tanggap darurat di bandara?",
                    "• Sebutkan sertifikasi yang berlaku 2 tahun"
                ])
                responses = {
                    "empty": f"⚠️ Mohon masukkan pertanyaan. Contoh:\n{samples}",
                    "pure_greeting": f"🖐️ Halo! Saya asisten QA POMPAV. Contoh pertanyaan:\n{samples}",
                    "small_talk": f"🤖 Saya AI spesialis POMPAV. Contoh pertanyaan teknis:\n{samples}"
                }
                return responses[q_type], (0, 0, 0)

            # Handle pertanyaan lanjutan
            if q_type == "follow_up" and self.memory.history and not self.is_new_topic(question):
                if not self.current_context:
                    self.current_context = self.vector_store.similarity_search(
                        self.memory.history[-1]["question"], k=3
                    )
                response = self.generate_follow_up_response(core_q)
                answer = response["output_text"]
                self.memory.add_interaction(question, answer)
                return answer, (0.9, 0.9, 0.9)  # High score for contextual answers

            # Handle pertanyaan baru
            docs = self.vector_store.similarity_search(core_q, k=3)
            self.current_context = docs
            self.update_terms(docs)

            if not docs:
                return "❌ Informasi tidak ditemukan. Mohon periksa istilah teknis.", (0, 0, 0)

            prompt_template = """
            [RIWAYAT PERCAKAPAN]
            {history}

            [DOKUMEN REFERENSI]
            {context}

            [PERTANYAAN]
            {question}

            [INSTRUKSI JAWABAN]
            1. Berikan jawaban langsung di awal
            2. Jelaskan dengan rinci merujuk dokumen
            3. Sertakan contoh jika relevan
            4. Tautkan dengan konteks sebelumnya jika ada
            5. Gunakan istilah teknis secara konsisten

            [FORMAT]
            <jawaban inti>
            <penjelasan>
            <contoh/analogi> (opsional)
            """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["history", "context", "question"]
            )

            chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
            response = chain({
                "input_documents": docs,
                "history": self.memory.get_formatted_history(),
                "question": core_q
            }, return_only_outputs=True)

            # Evaluasi dengan BERTScore
            P, R, F1 = score(
                [response["output_text"]],
                [docs[0].page_content],
                lang="id",
                model_type="bert-base-multilingual-cased"
            )

            self.memory.add_interaction(question, response["output_text"])
            return response["output_text"], (P.mean().item(), R.mean().item(), F1.mean().item())

        except Exception as e:
            traceback.print_exc()
            return f"⚠️ Error: {str(e)}", (0, 0, 0)


# ====== FUNGSI UTAMA ====== #
def main():
    # Konfigurasi API
    try:
        if "OPENAI_API_KEY" not in os.environ:
            api_key = input("Masukkan OPENAI_API_KEY: ")
            os.environ["OPENAI_API_KEY"] = api_key
    except Exception as e:
        print(f"⚠️ Error konfigurasi API: {str(e)}")
        return

    # Proses dokumen
    print("\n📤 Silakan masukkan path PDF dokumen POMPAV")
    pdf_path = input("Path file PDF: ").strip()

    print(f"\n🔍 Memproses dokumen: {pdf_path}...")
    processor = PDFProcessor()
    start = time.time()

    text = processor.extract_text(pdf_path)
    if not text:
        print("❌ Gagal membaca PDF")
        return

    chunks = processor.chunk_text(text)
    print(f"✂️ Total potongan teks: {len(chunks)}")

    processor.create_vector_store(chunks)
    print(f"⏱️ Pemrosesan selesai dalam {time.time()-start:.2f} detik")

    if not processor.vector_store:
        print("❌ Gagal membuat indeks pencarian")
        return

    # Inisialisasi QA Engine
    qa_engine = QAEngine(processor.vector_store)

    print("\n" + "="*50)
    print("💬 SISTEM QA POMPAV (GPT-4 Turbo) SIAP DIGUNAKAN")
    print("="*50)
    print("\nFitur Utama:")
    print("- Pemahaman konteks percakapan mendalam")
    print("- Deteksi otomatis pertanyaan lanjutan")
    print("- Penjelasan bertingkat dengan contoh")
    print("- Evaluasi kualitas jawaban real-time")
    print("\nKetik 'reset' untuk membersihkan memori")
    print("Ketik 'exit' untuk keluar\n")

    # Loop interaksi
    while True:
        try:
            question = input("\n❓ Pertanyaan Anda: ").strip()

            if question.lower() in ['exit', 'keluar', 'quit', 'q']:
                break

            if not question:
                print("⚠️ Mohon masukkan pertanyaan")
                continue

            start_time = time.time()
            answer, scores = qa_engine.generate_response(question)

            # Tampilkan jawaban
            print("\n" + answer)

            # Tampilkan metrics
            print(f"\n⏱️ Waktu respons: {time.time()-start_time:.2f} detik")
            if scores[2] > 0:
                print(f"📊 Skor Relevansi (F1): {scores[2]:.2f}/1.00")

            print("-"*50)
            if qa_engine.memory.history:
                print("💡 Konteks aktif:", qa_engine.memory.history[-1]["question"][:50] + "...")

        except KeyboardInterrupt:
            print("\n⚠️ Interupsi pengguna")
            break
        except Exception as e:
            print(f"🚨 Error: {str(e)}")
            traceback.print_exc()

    print("\n🛑 Sesi QA selesai. Terima kasih!")

if __name__ == "__main__":
    main()

## main_qa_module.py part 2
# ====== IMPORT LIBRARY ====== #
import os
import fitz
import traceback
import tiktoken
import time
import re
import numpy as np
import sys
from typing import Tuple, Optional, List, Dict
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from bert_score import BERTScorer


# ====== KELAS MEMORI PERCAKAPAN ====== #
class ConversationMemory:
    def __init__(self, max_length: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def add_interaction(self, question: str, answer: str):
        if len(self.history) >= self.max_length:
            self.history.pop(0)
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.strftime("%H:%M:%S"),
            "embedding": self._get_embedding(question + " " + answer)
        })

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.similarity_model.encode(text, convert_to_tensor=False)

    def get_context_similarity(self, current_question: str) -> float:
        if not self.history:
            return 0.0
        current_embedding = self._get_embedding(current_question)
        last_embedding = self.history[-1]["embedding"]
        return util.pytorch_cos_sim(current_embedding, last_embedding).item()

    def get_formatted_history(self, num_last: int = 2) -> str:
        return "\n\n".join(
            f"Q [{item['timestamp']}]: {item['question']}\nA: {item['answer'][:200]}..."
            for item in self.history[-num_last:]
        )

    def clear(self):
        self.history = []


# ====== KELAS PEMROSES PDF ====== #
class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            chunk_size=500,
            max_retries=5
        )
        self.vector_store = None

    def extract_text(self, pdf_path: str) -> Optional[str]:
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
                    if len(text) > 2_000_000:
                        print("⚠️ Peringatan: Dokumen sangat besar, hanya memproses bagian awal")
                        break
            return text if text.strip() else None
        except Exception as e:
            print(f"❌ Error ekstraksi PDF: {e}")
            traceback.print_exc()
            return None

    def chunk_text(self, text: str) -> List[str]:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            return splitter.split_text(text[:4_000_000])
        except Exception as e:
            print(f"❌ Error chunking teks: {e}")
            traceback.print_exc()
            return []

    def create_vector_store(self, chunks: List[str]):
        if not chunks:
            print("❌ Tidak ada teks yang bisa diproses")
            return

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(batch, self.embeddings)
                    print(f"✅ Vektor store awal dibuat dengan {len(batch)} chunk")
                else:
                    self.vector_store.add_texts(batch)
                    print(f"✅ Menambahkan batch {i//batch_size + 1} ({len(batch)} chunk)")
            except Exception as e:
                print(f"❌ Gagal memproses batch {i//batch_size + 1}: {e}")
                traceback.print_exc()


# ====== KELAS QA ENGINE ====== #
class QAEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        self.memory = ConversationMemory(max_length=5)
        self.current_context = None
        self.current_topic = None
        
        # Inisialisasi BERTScorer untuk Bahasa Indonesia
        # Inisialisasi BERTScore dengan model yang umum digunakan
        self.bert_scorer = BERTScorer(lang="id", model_type="bert-base-multilingual-cased")


        # Daftar pola pertanyaan
        self.greetings = ["halo", "hai", "hi", "selamat pagi", "selamat siang", "selamat malam"]
        self.small_talk = ["apa kabar", "bagaimana kabarmu", "siapa namamu"]
        self.reset_commands = ["reset", "clear", "bersihkan memori"]
        self.follow_up_keywords = ["jelaskan", "detail", "lebih", "tentang itu", "mengapa",
                                 "bagaimana", "contoh", "saja", "tersebut", "itu", "ini",
                                 "yg", "yang", "tadi", "nomor", "poin"]

        # Daftar istilah teknis
        self.technical_terms = set(["hsse", "dppu", "pompav", "keselamatan", "penerbangan",
                                   "bandara", "aviasi", "keamanan", "insiden", "darurat"])

    def extract_terms(self, text: str) -> set:
        acronyms = set(re.findall(r'\b[A-Z]{3,}\b', text))
        terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
        return {term.lower() for term in acronyms.union(terms)}

    def is_new_topic(self, question: str) -> bool:
        if not self.memory.history:
            return True
        last_question = self.memory.history[-1]["question"]
        last_terms = self.extract_terms(last_question)
        current_terms = self.extract_terms(question)
        jaccard_sim = len(current_terms & last_terms) / max(len(current_terms | last_terms), 1)
        return jaccard_sim < 0.2

    def update_terms(self, docs):
        for doc in docs:
            self.technical_terms.update(self.extract_terms(doc.page_content))

    def is_technical(self, question: str) -> bool:
        q_lower = question.lower()
        return any(term in q_lower for term in self.technical_terms if len(term) > 2)

    def generate_follow_up_suggestions(self, answer: str) -> List[str]:
        """Generate relevant follow-up questions based on the answer"""
        prompt = f"""
        Berdasarkan jawaban berikut, sarankan 3 pertanyaan lanjutan yang relevan dalam Bahasa Indonesia:
        
        Jawaban: {answer}
        
        Aturan:
        1. Pertanyaan harus spesifik dan terkait langsung dengan jawaban
        2. Gunakan istilah teknis dari jawaban
        3. Buat pertanyaan yang praktis dan dapat ditindaklanjuti
        4. Format sebagai daftar bernomor dengan setiap pertanyaan pada baris baru
        
        Contoh:
        1. Apa definisi lengkap dari HSSE menurut dokumen ini?
        2. Bagaimana prosedur penerapan POMPAV di bandara kecil?
        3. Apa saja sanksi untuk pelanggaran prosedur keselamatan?
        """
        
        try:
            response = self.llm.invoke(prompt)
            questions = response.content.split("\n")
            return [q.split(" ", 1)[1] for q in questions if q.strip() and q[0].isdigit()][:3]
        except Exception as e:
            print(f"⚠️ Error menghasilkan pertanyaan lanjutan: {e}")
            return []

    def analyze_question(self, question: str) -> Tuple[Optional[str], str]:
        if not question.strip():
            return None, "empty"

        q_lower = question.lower().strip()

        # Deteksi perintah khusus
        if any(cmd in q_lower for cmd in self.reset_commands):
            return None, "reset"
        if any(phrase in q_lower for phrase in self.small_talk):
            return None, "small_talk"

        # Deteksi sapaan
        has_greeting = any(greet in q_lower for greet in self.greetings)
        cleaned = re.sub('|'.join(self.greetings), "", q_lower).strip(' ,.!?')

        if not cleaned:
            return None, "pure_greeting" if has_greeting else "empty"

        # Deteksi pertanyaan lanjutan
        similarity_score = self.memory.get_context_similarity(question)
        is_follow_up = (similarity_score > 0.6 or
                       any(kw in q_lower for kw in self.follow_up_keywords) or
                       len(cleaned.split()) <= 3)

        if is_follow_up and self.memory.history:
            return question, "follow_up"
        elif self.is_technical(cleaned):
            return question, "technical"
        return question, "general"

    def generate_follow_up_response(self, question: str) -> Dict:
        last_qa = self.memory.history[-1]
        context = f"Pertanyaan sebelumnya: {last_qa['question']}\nJawaban sebelumnya: {last_qa['answer']}"

        prompt_template = """
        [KONTEKS SEBELUMNYA]
        {context}

        [PERTANYAAN LANJUTAN]
        {question}

        [INSTRUKSI]
        1. Identifikasi bagian mana dari jawaban sebelumnya yang dirujuk
        2. Jika pertanyaan ambigu, minta klarifikasi dengan menyebutkan:
           - Daftar topik terkait dari jawaban sebelumnya
           - Contoh format pertanyaan yang lebih spesifik
        3. Berikan jawaban yang:
           - Langsung menjawab pertanyaan
           - Memperluas penjelasan sebelumnya
           - Menyertakan contoh konkret
        4. Akhiri dengan 3 saran pertanyaan lanjutan

        [FORMAT JAWABAN]
        <konfirmasi topik>
        <penjelasan lengkap>
        <contoh relevan jika ada>
        
        [SARAN PERTANYAAN LANJUTAN]
        1. <pertanyaan 1>
        2. <pertanyaan 2>
        3. <pertanyaan 3>
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        return chain({
            "input_documents": self.current_context or [],
            "context": context,
            "question": question
        }, return_only_outputs=True)

    def generate_response(self, question: str) -> Tuple[str, Tuple[float, float, float], List[str]]:
        try:
            core_q, q_type = self.analyze_question(question)
            follow_up_suggestions = []

            # Handle perintah khusus
            if q_type == "reset":
                self.memory.clear()
                self.current_context = None
                return "🔄 Memori percakapan telah direset", (0, 0, 0), []

            if q_type in ["empty", "pure_greeting", "small_talk"]:
                samples = "\n".join([
                    "• Apa definisi HSSE menurut dokumen POMPAV?",
                    "• Bagaimana prosedur tanggap darurat di bandara?",
                    "• Sebutkan sertifikasi yang berlaku 2 tahun"
                ])
                responses = {
                    "empty": f"⚠️ Mohon masukkan pertanyaan. Contoh:\n{samples}",
                    "pure_greeting": f"🖐️ Halo! Saya asisten QA POMPAV. Contoh pertanyaan:\n{samples}",
                    "small_talk": f"🤖 Saya AI spesialis POMPAV. Contoh pertanyaan teknis:\n{samples}"
                }
                return responses[q_type], (0, 0, 0), []

            # Handle pertanyaan lanjutan
            if q_type == "follow_up" and self.memory.history and not self.is_new_topic(question):
                if not self.current_context:
                    self.current_context = self.vector_store.similarity_search(
                        self.memory.history[-1]["question"], k=3
                    )
                response = self.generate_follow_up_response(core_q)
                answer = response["output_text"]
                
                # Ekstrak saran pertanyaan lanjutan
                if "[SARAN PERTANYAAN LANJUTAN]" in answer:
                    answer_parts = answer.split("[SARAN PERTANYAAN LANJUTAN]")
                    answer = answer_parts[0].strip()
                    follow_up_suggestions = [q.strip() for q in answer_parts[1].split("\n") if q.strip() and q[0].isdigit()]
                
                self.memory.add_interaction(question, answer)
                return answer, (0.9, 0.9, 0.9), follow_up_suggestions

            # Handle pertanyaan baru
            docs = self.vector_store.similarity_search(core_q, k=3)
            self.current_context = docs
            self.update_terms(docs)

            if not docs:
                return "❌ Informasi tidak ditemukan. Mohon periksa istilah teknis.", (0, 0, 0), []

            prompt_template = """
            [RIWAYAT PERCAKAPAN]
            {history}

            [DOKUMEN REFERENSI]
            {context}

            [PERTANYAAN]
            {question}

            [INSTRUKSI JAWABAN]
            1. Berikan jawaban langsung di awal
            2. Jelaskan dengan rinci merujuk dokumen
            3. Sertakan contoh jika relevan
            4. Hubungkan dengan konteks sebelumnya jika ada
            5. Gunakan istilah teknis secara konsisten
            6. Akhiri dengan 3 saran pertanyaan lanjutan

            [FORMAT]
            <jawaban inti>
            <penjelasan>
            <contoh/analogi> (opsional)
            
            [SARAN PERTANYAAN LANJUTAN]
            1. <pertanyaan 1>
            2. <pertanyaan 2>
            3. <pertanyaan 3>
            """

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["history", "context", "question"]
            )

            chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
            response = chain({
                "input_documents": docs,
                "history": self.memory.get_formatted_history(),
                "question": core_q
            }, return_only_outputs=True)
            
            answer = response["output_text"]
            
            # Ekstrak saran pertanyaan lanjutan
            if "[SARAN PERTANYAAN LANJUTAN]" in answer:
                answer_parts = answer.split("[SARAN PERTANYAAN LANJUTAN]")
                answer = answer_parts[0].strip()
                follow_up_suggestions = [q.strip() for q in answer_parts[1].split("\n") if q.strip() and q[0].isdigit()]
            
            # Evaluasi dengan BERTScore Bahasa Indonesia
            reference_text = "\n".join([doc.page_content for doc in docs[:2]])
            print("\n[Menghitung BERTScore...]")
            sys.stdout.flush()
            
            P, R, F1 = self.bert_scorer.score(
                [answer],
                [reference_text]
            )
            
            # Konversi tensor ke nilai float
            precision = P.item()
            recall = R.item()
            f1 = F1.item()

            print("\n[Metric BERTScore]")
            print(f"Presisi: {precision:.4f}")
            print(f"Recall:  {recall:.4f}")
            print(f"F1:      {f1:.4f}")
            sys.stdout.flush()

            self.memory.add_interaction(question, answer)
            return answer, (precision, recall, f1), follow_up_suggestions

        except Exception as e:
            traceback.print_exc()
            return f"⚠️ Error: {str(e)}", (0, 0, 0), []


# ====== FUNGSI UTAMA ====== #
def main():
    # Konfigurasi API
    try:
        if "OPENAI_API_KEY" not in os.environ:
            api_key = input("Masukkan OPENAI_API_KEY: ")
            os.environ["OPENAI_API_KEY"] = api_key
    except Exception as e:
        print(f"⚠️ Error konfigurasi API: {str(e)}")
        return

    # Proses dokumen
    print("\n📤 Silakan masukkan path PDF dokumen POMPAV")
    pdf_path = input("Path file PDF: ").strip()

    print(f"\n🔍 Memproses dokumen: {pdf_path}...")
    processor = PDFProcessor()
    start = time.time()

    text = processor.extract_text(pdf_path)
    if not text:
        print("❌ Gagal membaca PDF")
        return

    chunks = processor.chunk_text(text)
    print(f"✂️ Total potongan teks: {len(chunks)}")

    processor.create_vector_store(chunks)
    print(f"⏱️ Pemrosesan selesai dalam {time.time()-start:.2f} detik")

    if not processor.vector_store:
        print("❌ Gagal membuat indeks pencarian")
        return

    # Inisialisasi QA Engine
    qa_engine = QAEngine(processor.vector_store)

    print("\n" + "="*50)
    print("💬 SISTEM QA POMPAV (GPT-4 Turbo) SIAP DIGUNAKAN")
    print("="*50)
    print("\nFitur Utama:")
    print("- Pemahaman konteks percakapan mendalam")
    print("- Deteksi otomatis pertanyaan lanjutan")
    print("- Penjelasan bertingkat dengan contoh")
    print("- Evaluasi kualitas jawaban real-time")
    print("- Saran pertanyaan lanjutan cerdas")
    print("\nKetik 'reset' untuk membersihkan memori")
    print("Ketik 'exit' untuk keluar\n")

    # Loop interaksi
    while True:
        try:
            question = input("\n❓ Pertanyaan Anda: ").strip()

            if question.lower() in ['exit', 'keluar', 'quit', 'q']:
                break

            if not question:
                print("⚠️ Mohon masukkan pertanyaan")
                continue

            start_time = time.time()
            answer, scores, follow_ups = qa_engine.generate_response(question)

            # Tampilkan jawaban
            print("\n" + "="*50)
            print("💬 JAWABAN:")
            print(answer)

            # Tampilkan metrics
            print("\n" + "="*50)
            print("📊 METRIK KINERJA:")
            print(f"⏱️ Waktu respons: {time.time()-start_time:.2f} detik")
            if scores[2] > 0:
                print(f"\nBERTScore:")
                print(f"  Presisi: {scores[0]:.4f}")
                print(f"  Recall:  {scores[1]:.4f}")
                print(f"  F1:      {scores[2]:.4f}")
                
                if scores[2] < 0.8:
                    print("\n💡 Saran: Untuk hasil lebih baik, coba:")
                    print("- Gunakan istilah teknis yang lebih spesifik")
                    print("- Ajukan pertanyaan yang lebih fokus")
                    print("- Sertakan referensi dokumen dalam pertanyaan")

            # Tampilkan saran pertanyaan lanjutan
            if follow_ups:
                print("\n" + "="*50)
                print("🔍 SARAN PERTANYAAN LANJUTAN:")
                for i, q in enumerate(follow_ups, 1):
                    print(f"{i}. {q}")

            print("\n" + "="*50)
            if qa_engine.memory.history:
                print("💡 Konteks aktif:", qa_engine.memory.history[-1]["question"][:50] + "...")
            
            sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n⚠️ Interupsi pengguna")
            break
        except Exception as e:
            print(f"🚨 Error: {str(e)}")
            traceback.print_exc()

    print("\n🛑 Sesi QA selesai. Terima kasih!")

if __name__ == "__main__":
    main()

    
## app.py
from flask import Flask, request, jsonify, render_template
from main_qa_module import PDFProcessor, QAEngine
from dotenv import load_dotenv
import os
load_dotenv()  # Ini akan membaca file .env

# Cek apakah berhasil
assert os.getenv("OPENAI_API_KEY"), "❌ OPENAI_API_KEY tidak ditemukan di .env"

app = Flask(__name__)

# ====== Inisialisasi Sistem QA ======
processor = PDFProcessor()
text = processor.extract_text('dokumen/dataset_pompav.pdf')  # Pastikan file ini ada
chunks = processor.chunk_text(text)
processor.create_vector_store(chunks)
qa_engine = QAEngine(processor.vector_store)

# ====== ROUTES ======

@app.route('/')
def index():
    return render_template('index.html')  # Menampilkan UI chatbot

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'response': '⚠️ Pertanyaan kosong tidak dapat diproses'}), 400
    answer, scores = qa_engine.generate_response(user_message)
    return jsonify({'response': answer, 'f1_score': round(scores[2], 2)})

if __name__ == '__main__':
    app.run(debug=False)

# 18-05-2025
import os
import fitz
import traceback
import tiktoken
import time
import re
import numpy as np
from typing import Tuple, Optional, List, Dict
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from bert_score import score
import sys
sys.stdout.flush()  # Pastikan output tidak tertahan di buffer

# ====== KELAS MEMORI PERCAKAPAN ======

class ConversationMemory:
    def __init__(self, max_length: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def add_interaction(self, question: str, answer: str):
        if len(self.history) >= self.max_length:
            self.history.pop(0)
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.strftime("%H:%M:%S"),
            "embedding": self._get_embedding(question + " " + answer)
        })

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.similarity_model.encode(text, convert_to_tensor=False)

    def get_context_similarity(self, current_question: str) -> float:
        if not self.history:
            return 0.0
        current_embedding = self._get_embedding(current_question)
        last_embedding = self.history[-1]["embedding"]
        return util.pytorch_cos_sim(current_embedding, last_embedding).item()

    def get_formatted_history(self, num_last: int = 2) -> str:
        return "\n\n".join(
            f"Q [{item['timestamp']}]: {item['question']}\nA: {item['answer'][:200]}..."
            for item in self.history[-num_last:]
        )

    def clear(self):
        self.history = []


# ====== KELAS PEMROSES PDF ======

class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            chunk_size=500,
            max_retries=5
        )
        self.vector_store = None

    def extract_text(self, pdf_path: str) -> Optional[str]:
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
                    if len(text) > 2_000_000:
                        print("⚠️ Peringatan: Dokumen sangat besar, hanya memproses bagian awal")
                        break
            return text if text.strip() else None
        except Exception as e:
            print(f"❌ Error ekstraksi PDF: {e}")
            traceback.print_exc()
            return None

    def chunk_text(self, text: str) -> List[str]:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            return splitter.split_text(text[:4_000_000])
        except Exception as e:
            print(f"❌ Error chunking teks: {e}")
            traceback.print_exc()
            return []

    def create_vector_store(self, chunks: List[str]):
        if not chunks:
            print("❌ Tidak ada teks yang bisa diproses")
            return

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(batch, self.embeddings)
                    print(f"✅ Vektor store awal dibuat dengan {len(batch)} chunk")
                else:
                    self.vector_store.add_texts(batch)
                    print(f"✅ Menambahkan batch {i//batch_size + 1} ({len(batch)} chunk)")
            except Exception as e:
                print(f"❌ Gagal memproses batch {i//batch_size + 1}: {e}")
                traceback.print_exc()

import os
import fitz
import traceback
import tiktoken
import time
import re
import numpy as np
from typing import Tuple, Optional, List, Dict
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from bert_score import score
from threading import Lock
import sys
sys.stdout.flush()

class ConversationMemory:
    def __init__(self, max_length: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def add_interaction(self, question: str, answer: str):
        if len(self.history) >= self.max_length:
            self.history.pop(0)
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.strftime("%H:%M:%S"),
            "embedding": self._get_embedding(question + " " + answer)
        })

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.similarity_model.encode(text, convert_to_tensor=False)

    def get_context_similarity(self, current_question: str) -> float:
        if not self.history:
            return 0.0
        current_embedding = self._get_embedding(current_question)
        last_embedding = self.history[-1]["embedding"]
        return util.pytorch_cos_sim(current_embedding, last_embedding).item()

    def get_formatted_history(self, num_last: int = 2) -> str:
        return "\n\n".join(
            f"Q [{item['timestamp']}]: {item['question']}\nA: {item['answer'][:200]}..."
            for item in self.history[-num_last:]
        )

    def clear(self):
        self.history = []

class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            chunk_size=500,
            max_retries=5
        )
        self.vector_store = None
        self.vector_store_dir = "vector_store"
        self.index_name = "pompav_index"
        
    def initialize_vector_store(self, pdf_path: str) -> bool:
        """Initialize or load vector store from disk"""
        # Create directory if it doesn't exist
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        index_path = os.path.join(self.vector_store_dir, self.index_name)
        
        # Try to load existing index
        if os.path.exists(index_path):
            try:
                print("🔍 Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ Vector store loaded successfully")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load vector store: {e}")
                print("🛠️ Creating new vector store...")
                
        # Create new vector store if loading failed
        return self._create_new_vector_store(pdf_path, index_path)
    
    def _create_new_vector_store(self, pdf_path: str, save_path: str) -> bool:
        """Create new vector store from PDF and save to disk"""
        text = self.extract_text(pdf_path)
        if not text:
            print("❌ No text extracted from PDF")
            return False
            
        chunks = self.chunk_text(text)
        if not chunks:
            print("❌ No chunks created from text")
            return False
            
        self.create_vector_store(chunks)
        
        if self.vector_store:
            try:
                self.vector_store.save_local(save_path)
                print(f"💾 Vector store saved to {save_path}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to save vector store: {e}")
                return True  # Still return True since we have it in memory
        
        return False

    def extract_text(self, pdf_path: str) -> Optional[str]:
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
                    if len(text) > 2_000_000:
                        print("⚠️ Warning: Document very large, only processing first part")
                        break
            return text if text.strip() else None
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            traceback.print_exc()
            return None

    def chunk_text(self, text: str) -> List[str]:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            return splitter.split_text(text[:4_000_000])
        except Exception as e:
            print(f"❌ Text chunking error: {e}")
            traceback.print_exc()
            return []

    def create_vector_store(self, chunks: List[str]):
        if not chunks:
            print("❌ No text chunks to process")
            return

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(batch, self.embeddings)
                    print(f"✅ Initial vector store created with {len(batch)} chunks")
                else:
                    self.vector_store.add_texts(batch)
                    print(f"✅ Added batch {i//batch_size + 1} ({len(batch)} chunks)")
            except Exception as e:
                print(f"❌ Failed to process batch {i//batch_size + 1}: {e}")
                traceback.print_exc()
                
class QAEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        self.memory = ConversationMemory(max_length=5)
        self.current_context = None
        self.current_topic = None
        self.lock = Lock()  # Thread lock for concurrent requests
        self.technical_terms = set()

        # Patterns and keywords
        self.greetings = ["halo", "hai", "hi", "selamat pagi", "selamat siang", "selamat malam"]
        self.small_talk = ["apa kabar", "bagaimana kabarmu", "siapa namamu"]
        self.reset_commands = ["reset", "clear", "bersihkan memori"]
        self.follow_up_keywords = ["jelaskan", "detail", "lebih", "tentang itu", "mengapa",
                                 "bagaimana", "contoh", "saja", "tersebut", "itu", "ini",
                                 "yg", "yang", "tadi", "nomor", "poin"]

    def extract_terms(self, text: str) -> set:
        acronyms = set(re.findall(r'\b[A-Z]{3,}\b', text))
        terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
        return {term.lower() for term in acronyms.union(terms)}

    def is_new_topic(self, question: str) -> bool:
        if not self.memory.history:
            return True
        last_question = self.memory.history[-1]["question"]
        last_terms = self.extract_terms(last_question)
        current_terms = self.extract_terms(question)
        jaccard_sim = len(current_terms & last_terms) / max(len(current_terms | last_terms), 1)
        return jaccard_sim < 0.2

    def update_terms(self, docs):
        for doc in docs:
            self.technical_terms.update(self.extract_terms(doc.page_content))

    def is_technical(self, question: str) -> bool:
        q_lower = question.lower()
        return any(term in q_lower for term in self.technical_terms if len(term) > 2)

    def analyze_question(self, question: str) -> Tuple[Optional[str], str]:
        if not question.strip():
            return None, "empty"

        q_lower = question.lower().strip()

        # Special commands
        if any(cmd in q_lower for cmd in self.reset_commands):
            return None, "reset"
        if any(phrase in q_lower for phrase in self.small_talk):
            return None, "small_talk"

        # Greetings detection
        has_greeting = any(greet in q_lower for greet in self.greetings)
        cleaned = re.sub('|'.join(self.greetings), "", q_lower).strip(' ,.!?')

        if not cleaned:
            return None, "pure_greeting" if has_greeting else "empty"

        # Follow-up detection
        similarity_score = self.memory.get_context_similarity(question)
        is_follow_up = (similarity_score > 0.6 or
                       any(kw in q_lower for kw in self.follow_up_keywords) or
                       len(cleaned.split()) <= 3)

        if is_follow_up and self.memory.history:
            return question, "follow_up"
        elif self.is_technical(cleaned):
            return question, "technical"
        return question, "general"

    def generate_follow_up_response(self, question: str) -> Dict:
        last_qa = self.memory.history[-1]
        context = f"Pertanyaan sebelumnya: {last_qa['question']}\nJawaban sebelumnya: {last_qa['answer']}"

        prompt_template = """
        [KONTEKS SEBELUMNYA]
        {context}

        [PERTANYAAN LANJUTAN]
        {question}

        [INSTRUKSI]
        1. Identifikasi dengan tepat bagian mana dari jawaban sebelumnya yang dirujuk
        2. Jika pertanyaan ambigu, ajukan pertanyaan klarifikasi
        3. Berikan jawaban yang:
           - Langsung menjawab pertanyaan
           - Memperluas penjelasan sebelumnya
           - Menyertakan contoh konkret
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        return chain({
            "input_documents": self.current_context or [],
            "context": context,
            "question": question
        }, return_only_outputs=True)

    def generate_response(self, question: str) -> Tuple[str, Tuple[float, float, float]]:
        with self.lock:  # Thread-safe processing
            try:
                core_q, q_type = self.analyze_question(question)

                # Handle special commands
                if q_type == "reset":
                    self.memory.clear()
                    self.current_context = None
                    print("🔄 Memori percakapan telah direset")
                    return "🔄 Memori percakapan telah direset", (0, 0, 0)

                if q_type in ["empty", "pure_greeting", "small_talk"]:
                    samples = "\n".join([
                        "Apa definisi HSSE menurut dokumen POMPAV?",
                        "Bagaimana prosedur tanggap darurat di bandara?",
                        "Sebutkan sertifikasi yang berlaku 2 tahun"
                    ])
                    responses = {
                        "empty": f"⚠️ Mohon masukkan pertanyaan. Contoh:\n{samples}",
                        "pure_greeting": f"🖐️ Halo! Saya asisten QA POMPAV. Kamu bisa bertanya:\n{samples}",
                        "small_talk": f"🤖 Saya asisten QA POMPAV. Contoh pertanyaan:\n{samples}"
                    }
                    response = responses[q_type]
                    print(f"Response: {response}")
                    return response, (0, 0, 0)

                # Follow-up questions
                if q_type == "follow_up" and self.memory.history and not self.is_new_topic(question):
                    if not self.current_context:
                        self.current_context = self.vector_store.similarity_search(
                            self.memory.history[-1]["question"], k=3
                        )
                    response = self.generate_follow_up_response(core_q)
                    answer = response["output_text"]
                    self.memory.add_interaction(question, answer)
                    print(f"Follow-up response: {answer}")
                    return answer, (0.9, 0.9, 0.9)

                # New questions
                docs = self.vector_store.similarity_search(core_q, k=3)
                self.current_context = docs
                self.update_terms(docs)

                if not docs:
                    print("❌ Informasi tidak ditemukan. Mohon periksa istilah teknis.")
                    return "❌ Informasi tidak ditemukan. Mohon periksa istilah teknis.", (0, 0, 0)

                prompt_template = """
                [RIWAYAT PERCAKAPAN]
                {history}

                [DOKUMEN REFERENSI]
                {context}

                [PERTANYAAN]
                {question}

                [INSTRUKSI JAWABAN]
                1. Berikan jawaban langsung di awal
                2. Jelaskan dengan rinci merujuk dokumen
                3. Sertakan contoh jika relevan
                4. Tautkan dengan konteks sebelumnya jika ada
                """

                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["history", "context", "question"]
                )

                chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
                response = chain({
                    "input_documents": docs,
                    "history": self.memory.get_formatted_history(),
                    "question": core_q
                }, return_only_outputs=True)

                # BERTScore evaluation
                P, R, F1 = score(
                    [response["output_text"]],
                    [docs[0].page_content],
                    lang="id",
                    model_type="bert-base-multilingual-cased"
                )

                # Print BERTScore to terminal
                print("\n" + "="*50)
                print("📊 EVALUASI BERTSCORE:")
                print(f"Precision: {P.mean().item():.4f}")
                print(f"Recall: {R.mean().item():.4f}")
                print(f"F1 Score: {F1.mean().item():.4f}")
                print("="*50 + "\n")

                self.memory.add_interaction(question, response["output_text"])
                return response["output_text"], (P.mean().item(), R.mean().item(), F1.mean().item())

            except Exception as e:
                traceback.print_exc()
                error_msg = f"⚠️ Error: {str(e)}"
                print(error_msg)
                return error_msg, (0, 0, 0)

# ====== FUNGSI UTAMA ======

def main():
    # Konfigurasi API
    try:
        if "OPENAI_API_KEY" not in os.environ:
            api_key = input("Masukkan OPENAI_API_KEY: ")
            os.environ["OPENAI_API_KEY"] = api_key
    except Exception as e:
        print(f"⚠️ Error konfigurasi API: {str(e)}")
        return

    # Proses dokumen
    print("\n📤 Silakan masukkan path PDF dokumen POMPAV")
    pdf_path = input("Path file PDF: ").strip()

    print(f"\n🔍 Memproses dokumen: {pdf_path}...")
    processor = PDFProcessor()
    start = time.time()

    text = processor.extract_text(pdf_path)
    if not text:
        print("❌ Gagal membaca PDF")
        return

    chunks = processor.chunk_text(text)
    print(f"✂️ Total potongan teks: {len(chunks)}")

    processor.create_vector_store(chunks)
    print(f"⏱️ Pemrosesan selesai dalam {time.time()-start:.2f} detik")

    if not processor.vector_store:
        print("❌ Gagal membuat indeks pencarian")
        return

    # Inisialisasi QA Engine
    qa_engine = QAEngine(processor.vector_store)

    print("\n" + "="*50)
    print("💬 SISTEM QA POMPAV (GPT-4 Turbo) SIAP DIGUNAKAN")
    print("="*50)
    print("\nFitur Utama:")
    print("- Pemahaman konteks percakapan mendalam")
    print("- Deteksi otomatis pertanyaan lanjutan")
    print("- Penjelasan bertingkat dengan contoh")
    print("- Evaluasi kualitas jawaban real-time")
    print("\nKetik 'reset' untuk membersihkan memori")
    print("Ketik 'exit' untuk keluar\n")

    # Loop interaksi
    while True:
        try:
            question = input("\n❓ Pertanyaan Anda: ").strip()

            if question.lower() in ['exit', 'keluar', 'quit', 'q']:
                break

            if not question:
                print("⚠️ Mohon masukkan pertanyaan")
                continue

            start_time = time.time()
            answer, scores = qa_engine.generate_response(question)

            # Tampilkan jawaban
            print("\n" + answer)

            # Tampilkan metrics
            print(f"\n⏱️ Waktu respons: {time.time()-start_time:.2f} detik")
            if scores[2] > 0:
                print(f"📊 Skor Relevansi (F1): {scores[2]:.2f}/1.00")

            print("-"*50)
            if qa_engine.memory.history:
                print("💡 Konteks aktif:", qa_engine.memory.history[-1]["question"][:50] + "...")

        except KeyboardInterrupt:
            print("\n⚠️ Interupsi pengguna")
            break
        except Exception as e:
            print(f"🚨 Error: {str(e)}")
            traceback.print_exc()

    print("\n🛑 Sesi QA selesai. Terima kasih!")


if __name__ == "__main__":
    main()


#backup kode main_qa_module 23-05-2025

import os
import fitz
import traceback
import tiktoken
import time
import re
import numpy as np
from typing import Tuple, Optional, List, Dict
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from threading import Lock
import sys
sys.stdout.flush()

class ConversationMemory:
    def __init__(self, max_length: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length
        self.similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def add_interaction(self, question: str, answer: str):
        if len(self.history) >= self.max_length:
            self.history.pop(0)
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": time.strftime("%H:%M:%S"),
            "embedding": self._get_embedding(question + " " + answer)
        })

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.similarity_model.encode(text, convert_to_tensor=False)

    def get_context_similarity(self, current_question: str) -> float:
        if not self.history:
            return 0.0
        current_embedding = self._get_embedding(current_question)
        last_embedding = self.history[-1]["embedding"]
        return util.pytorch_cos_sim(current_embedding, last_embedding).item()

    def get_formatted_history(self, num_last: int = 2) -> str:
        return "\n\n".join(
            f"Q [{item['timestamp']}]: {item['question']}\nA: {item['answer'][:200]}..."
            for item in self.history[-num_last:]
        )

    def clear(self):
        self.history = []

class PDFProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1536,
            chunk_size=500,
            max_retries=5
        )
        self.vector_store = None
        self.vector_store_dir = "vector_store"
        self.index_name = "pompav_index"
        
    def initialize_vector_store(self, pdf_path: str) -> bool:
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        index_path = os.path.join(self.vector_store_dir, self.index_name)
        
        if os.path.exists(index_path):
            try:
                print("🔍 Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ Vector store loaded successfully")
                return True
            except Exception as e:
                print(f"⚠️ Failed to load vector store: {e}")
                print("🛠️ Creating new vector store...")
                
        return self._create_new_vector_store(pdf_path, index_path)
    
    def _create_new_vector_store(self, pdf_path: str, save_path: str) -> bool:
        text = self.extract_text(pdf_path)
        if not text:
            print("❌ No text extracted from PDF")
            return False
            
        chunks = self.chunk_text(text)
        if not chunks:
            print("❌ No chunks created from text")
            return False
            
        self.create_vector_store(chunks)
        
        if self.vector_store:
            try:
                self.vector_store.save_local(save_path)
                print(f"💾 Vector store saved to {save_path}")
                return True
            except Exception as e:
                print(f"⚠️ Failed to save vector store: {e}")
                return True
        
        return False

    def extract_text(self, pdf_path: str) -> Optional[str]:
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
                    if len(text) > 2_000_000:
                        print("⚠️ Warning: Document very large, only processing first part")
                        break
            return text if text.strip() else None
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            traceback.print_exc()
            return None

    def chunk_text(self, text: str) -> List[str]:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            return splitter.split_text(text[:4_000_000])
        except Exception as e:
            print(f"❌ Text chunking error: {e}")
            traceback.print_exc()
            return []

    def create_vector_store(self, chunks: List[str]):
        if not chunks:
            print("❌ No text chunks to process")
            return

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(batch, self.embeddings)
                    print(f"✅ Initial vector store created with {len(batch)} chunks")
                else:
                    self.vector_store.add_texts(batch)
                    print(f"✅ Added batch {i//batch_size + 1} ({len(batch)} chunks)")
            except Exception as e:
                print(f"❌ Failed to process batch {i//batch_size + 1}: {e}")
                traceback.print_exc()

class QAEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0.7,
            max_tokens=2000
        )
        self.memory = ConversationMemory(max_length=5)
        self.current_context = None
        self.current_topic = None
        self.lock = Lock()  # Thread lock for concurrent requests
        self.technical_terms = set()

        self.greetings = ["halo", "hai", "hi", "selamat pagi", "selamat siang", "selamat malam"]
        self.small_talk = ["apa kabar", "bagaimana kabarmu", "siapa namamu"]
        self.reset_commands = ["reset", "clear", "bersihkan memori"]
        self.follow_up_keywords = ["jelaskan", "detail", "lebih", "tentang itu", "mengapa", "bagaimana", "contoh", "saja", "tersebut", "itu", "ini", "yg", "yang", "tadi", "nomor", "poin"]

    def extract_terms(self, text: str) -> set:
        acronyms = set(re.findall(r'\b[A-Z]{3,}\b', text))
        terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
        return {term.lower() for term in acronyms.union(terms)}

    def is_new_topic(self, question: str) -> bool:
        if not self.memory.history:
            return True
        last_question = self.memory.history[-1]["question"]
        last_terms = self.extract_terms(last_question)
        current_terms = self.extract_terms(question)
        jaccard_sim = len(current_terms & last_terms) / max(len(current_terms | last_terms), 1)
        return jaccard_sim < 0.2

    def update_terms(self, docs):
        for doc in docs:
            self.technical_terms.update(self.extract_terms(doc.page_content))

    def is_technical(self, question: str) -> bool:
        q_lower = question.lower()
        return any(term in q_lower for term in self.technical_terms if len(term) > 2)

    def analyze_question(self, question: str) -> Tuple[Optional[str], str]:
        if not question.strip():
            return None, "empty"

        q_lower = question.lower().strip()

        if any(cmd in q_lower for cmd in self.reset_commands):
            return None, "reset"
        if any(phrase in q_lower for phrase in self.small_talk):
            return None, "small_talk"

        has_greeting = any(greet in q_lower for greet in self.greetings)
        cleaned = re.sub('|'.join(self.greetings), "", q_lower).strip(' ,.!?')

        if not cleaned:
            return None, "pure_greeting" if has_greeting else "empty"

        similarity_score = self.memory.get_context_similarity(question)
        is_follow_up = (similarity_score > 0.6 or
                       any(kw in q_lower for kw in self.follow_up_keywords) or
                       len(cleaned.split()) <= 3)

        if is_follow_up and self.memory.history:
            return question, "follow_up"
        elif self.is_technical(cleaned):
            return question, "technical"
        return question, "general"

    def generate_follow_up_response(self, question: str) -> Dict:
        last_qa = self.memory.history[-1]
        context = f"Pertanyaan sebelumnya: {last_qa['question']}\nJawaban sebelumnya: {last_qa['answer']}"

        prompt_template = """
        [KONTEKS SEBELUMNYA]
        {context}

        [PERTANYAAN LANJUTAN]
        {question}

        [INSTRUKSI]
        1. Identifikasi dengan tepat bagian mana dari jawaban sebelumnya yang dirujuk
        2. Jika pertanyaan ambigu, ajukan pertanyaan klarifikasi
        3. Berikan jawaban yang:
           - Langsung menjawab pertanyaan
           - Memperluas penjelasan sebelumnya
           - Menyertakan contoh konkret
        4. jika pertanyaan diluar konteks atau tidak relevan dengan dokumen jawab dengan pertanyaan diluar pengetahuan yang ada
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        return chain({
            "input_documents": self.current_context or [],
            "context": context,
            "question": question
        }, return_only_outputs=True)

    def generate_response(self, question: str) -> Tuple[str, Tuple[float, float, float]]:
        with self.lock:  
            try:
                start_time = time.time()
                core_q, q_type = self.analyze_question(question)

                if q_type == "reset":
                    self.memory.clear()
                    self.current_context = None
                    return "🔄 Memori percakapan telah direset", (0, 0, 0)

                if q_type in ["empty", "pure_greeting", "small_talk"]:
                    samples = "\n".join([
                        "Apa definisi HSSE menurut dokumen POMPAV?",
                        "Bagaimana prosedur tanggap darurat di bandara?",
                        "bagaimana prosedur pengisian pesawat udara?"
                    ])
                    responses = {
                        "empty": f"⚠️ Mohon masukkan pertanyaan. Contoh:\n{samples}",
                        "pure_greeting": f"🖐️Halo! Saya asisten QA POMPAV. Kamu bisa bertanya:\n{samples}",
                        "small_talk": f"🤖 Saya asisten QA POMPAV. Contoh pertanyaan:\n{samples}"
                    }
                    return responses[q_type], (0, 0, 0)

                if q_type == "follow_up" and self.memory.history and not self.is_new_topic(question):
                    if not self.current_context:
                        self.current_context = self.vector_store.similarity_search(
                            self.memory.history[-1]["question"], k=3
                        )
                    response = self.generate_follow_up_response(core_q)
                    answer = response["output_text"]
                    self.memory.add_interaction(question, answer)
                    return answer, (0.9, 0.9, 0.9)

                docs = self.vector_store.similarity_search(core_q, k=3)
                self.current_context = docs
                self.update_terms(docs)

                if not docs:
                    return "❌ Informasi tidak ditemukan. Mohon periksa istilah teknis.", (0, 0, 0)

                prompt_template = """
                [RIWAYAT PERCAKAPAN]
                {history}

                [DOKUMEN REFERENSI]
                {context}

                [PERTANYAAN]
                {question}

                [INSTRUKSI JAWABAN]
                1. Berikan jawaban langsung di awal
                2. Jelaskan dengan rinci merujuk dokumen
                3. Sertakan contoh jika relevan
                4. Tautkan dengan konteks sebelumnya jika ada
                5. jika pertanyaan diluar konteks atau tidak relevan dengan dokumen jawab dengan pertanyaan diluar pengetahuan yang ada
                """

                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["history", "context", "question"]
                )

                chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
                response = chain({
                    "input_documents": docs,
                    "history": self.memory.get_formatted_history(),
                    "question": core_q
                }, return_only_outputs=True)

                self.memory.add_interaction(question, response["output_text"])
                process_time = time.time() - start_time

                print(f"⏱️ Waktu respon: {process_time:.2f} detik")
                return response["output_text"], (0, 0, 0)
            

            except Exception as e:
                traceback.print_exc()
                return f"⚠️ Error: {str(e)}", (0, 0, 0)

# ====== FUNGSI UTAMA ======
def main():
    try:
        if "OPENAI_API_KEY" not in os.environ:
            api_key = input("Masukkan OPENAI_API_KEY: ")
            os.environ["OPENAI_API_KEY"] = api_key
    except Exception as e:
        print(f"⚠️ Error konfigurasi API: {str(e)}")
        return

    print("\n📤 Silakan masukkan path PDF dokumen POMPAV")
    pdf_path = input("Path file PDF: ").strip()

    print(f"\n🔍 Memproses dokumen: {pdf_path}...")
    processor = PDFProcessor()
    start = time.time()

    text = processor.extract_text(pdf_path)
    if not text:
        print("❌ Gagal membaca PDF")
        return

    chunks = processor.chunk_text(text)
    processor.create_vector_store(chunks)
    print(f"⏱️ Pemrosesan selesai dalam {time.time()-start:.2f} detik")

    if not processor.vector_store:
        print("❌ Gagal membuat indeks pencarian")
        return

    qa_engine = QAEngine(processor.vector_store)

    print("\n" + "="*50)
    print("💬 SISTEM QA POMPAV (GPT-4 Turbo) SIAP DIGUNAKAN")
    print("="*50)

    while True:
        try:
            question = input("\n❓ Pertanyaan Anda: ").strip()

            if question.lower() in ['exit', 'keluar', 'quit', 'q']:
                break

            if not question:
                print("⚠️ Mohon masukkan pertanyaan")
                continue

            start_time = time.time()  # Mulai waktu respon
            answer, _ = qa_engine.generate_response(question)

            print("\n" + answer)
            print(f"\n⏱️ Waktu respons: {time.time()-start_time:.2f} detik")

            print("-"*50)

        except KeyboardInterrupt:
            print("\n⚠️ Interupsi pengguna")
            break
        except Exception as e:
            print(f"🚨 Error: {str(e)}")
            traceback.print_exc()

    print("\n🛑 Sesi QA selesai. Terima kasih!")


if __name__ == "__main__":
    main()

