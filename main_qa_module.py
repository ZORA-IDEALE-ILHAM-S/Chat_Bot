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
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import LLMChain, StuffDocumentsChain
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
                print("🔍 Memuat vector store yang sudah ada...")
                self.vector_store = FAISS.load_local(
                    folder_path=index_path,
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ Vector store berhasil dimuat")
                return True
            except Exception as e:
                print(f"⚠️ Gagal memuat vector store: {e}")
                print("🛠️ Membuat vector store baru...")
                
        return self._create_new_vector_store(pdf_path, index_path)
    
    def _create_new_vector_store(self, pdf_path: str, save_path: str) -> bool:
        text = self.extract_text(pdf_path)
        if not text:
            print("❌ Tidak ada teks yang diekstraksi dari PDF")
            return False
            
        chunks = self.chunk_text(text)
        if not chunks:
            print("❌ Tidak ada chunks yang dibuat dari teks")
            return False
            
        self.create_vector_store(chunks)
        
        if self.vector_store:
            try:
                self.vector_store.save_local(save_path)
                print(f"💾 Vector store disimpan ke {save_path}")
                return True
            except Exception as e:
                print(f"⚠️ Gagal menyimpan vector store: {e}")
                return True
        
        return False

    def extract_text(self, pdf_path: str) -> Optional[str]:
        text = ""
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
                    if len(text) > 1_000_000:
                        print("⚠️ Peringatan: Dokumen sangat besar, hanya memproses bagian pertama")
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
            print(f"❌ Error pembuatan chunks: {e}")
            traceback.print_exc()
            return []

    def create_vector_store(self, chunks: List[str]):
        if not chunks:
            print("❌ Tidak ada chunks teks untuk diproses")
            return

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                if not self.vector_store:
                    self.vector_store = FAISS.from_texts(batch, self.embeddings)
                    print(f"✅ Vector store awal dibuat dengan {len(batch)} chunks")
                else:
                    self.vector_store.add_texts(batch)
                    print(f"✅ Menambahkan batch {i//batch_size + 1} ({len(batch)} chunks)")
            except Exception as e:
                print(f"❌ Gagal memproses batch {i//batch_size + 1}: {e}")
                traceback.print_exc()

class QAEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000
        )
        self.memory = ConversationMemory(max_length=5)
        self.current_context = None
        self.main_topic = None
        self.main_topic_embedding = None
        self.topic_threshold = 0.5
        self.lock = Lock()
        self.technical_terms = set()
        
        self.greetings = ["halo", "hai", "hi", "selamat pagi", "selamat siang", "selamat malam"]
        self.small_talk = ["apa kabar", "bagaimana kabarmu", "siapa namamu"]
        self.reset_commands = ["reset", "clear", "bersihkan memori"]
        self.follow_up_keywords = ["jelaskan", "detail", "lebih", "tentang itu", "mengapa", "bagaimana", "contoh", "saja", "tersebut", "itu", "ini", "yg", "yang", "tadi", "nomor", "poin"]
        self.last_topic = None
        self.last_answer = None
    def extract_terms(self, text: str) -> set:
        acronyms = set(re.findall(r'\b[A-Z]{3,}\b', text))
        terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text))
        return {term.lower() for term in acronyms.union(terms)}

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.memory._get_embedding(text)

    def determine_main_topic(self, question: str):
        question_embedding = self._get_embedding(question)
        
        if self.main_topic is None:
            self.main_topic = question
            self.main_topic_embedding = question_embedding
            return
        
        similarity = util.pytorch_cos_sim(
            self.main_topic_embedding, 
            question_embedding
        ).item()
        
        if similarity > self.topic_threshold:
            return
        else:
            self.main_topic = question
            self.main_topic_embedding = question_embedding
            self.current_context = None

    def is_question_related_to_docs(self, question: str) -> bool:
        docs = self.vector_store.similarity_search(question, k=1)
        if not docs:
            return False
            
        doc_embedding = self._get_embedding(docs[0].page_content)
        question_embedding = self._get_embedding(question)
        similarity = util.pytorch_cos_sim(doc_embedding, question_embedding).item()
        
        return similarity > 0.6

    def is_new_topic(self, question: str) -> bool:
        if not self.memory.history:
            return True
            
        last_qa = self.memory.history[-1]
        last_text = f"{last_qa['question']} {last_qa['answer']}"
        current_text = question
        
        similarity = util.pytorch_cos_sim(
            self._get_embedding(last_text),
            self._get_embedding(current_text)
        ).item()
        
        return similarity < 0.4

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
        last_terms = self.extract_terms(self.memory.history[-1]["question"]) if self.memory.history else set()
        current_terms = self.extract_terms(question)
        
        is_follow_up = (
            similarity_score > 0.4 or
            any(kw in q_lower for kw in self.follow_up_keywords) or
            any(term in q_lower for term in last_terms) or
            not self.is_new_topic(question)
        )

        if is_follow_up and self.memory.history:
            return question, "follow_up"
        elif self.is_technical(cleaned):
            return question, "technical"
        return question, "general"

    def generate_follow_up_response(self, question: str) -> str:
        last_qa = self.memory.history[-1]
        context = f"Pertanyaan sebelumnya: {last_qa['question']}\nJawaban sebelumnya: {last_qa['answer']}"

        prompt_template = """
        [KONTEKS SEBELUMNYA]
        {context}

        [DOKUMEN REFERENSI]
        {documents}

        [PERTANYAAN LANJUTAN]
        {question}

        [INSTRUKSI]
        1. Identifikasi dengan tepat bagian mana dari jawaban sebelumnya yang dirujuk
        2. Jika pertanyaan ambigu, ajukan pertanyaan klarifikasi
        3. Berikan jawaban yang:
           - Lebih mendalam dari jawaban sebelumnya
           - Tetap relevan dengan konteks percakapan
           - Menggunakan dokumen referensi jika tersedia
        4. Jika pertanyaan di luar konteks, beri tahu dengan sopan
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "documents", "question"]
        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        return llm_chain.run(
            context=context,
            documents="\n\n".join([doc.page_content for doc in self.current_context]) if self.current_context else "Tidak ada dokumen referensi",
            question=question
        )
    
    def generate_response(self, question: str) -> Tuple[str, Tuple[float, float, float]]:
        with self.lock:  
            try:
                start_time = time.time()
                core_q, q_type = self.analyze_question(question)
                
                # Handle reset commands
                if q_type == "reset":
                    self.memory.clear()
                    self.current_context = None
                    self.last_topic = None
                    self.last_answer = None
                    return "🔄 Memori percakapan telah direset", (0, 0, 0)

                # Handle small talk and empty questions
                if q_type in ["empty", "pure_greeting", "small_talk"]:
                    samples = "\n".join([
                        "Apa definisi HSSE menurut dokumen POMPAV?",
                        "Bagaimana prosedur tanggap darurat di bandara?",
                        "Bagaimana prosedur pengisian pesawat udara?"
                    ])
                    responses = {
                        "empty": f"⚠️ Mohon masukkan pertanyaan. Contoh:\n{samples}",
                        "pure_greeting": f"🖐️ Halo! Saya asisten QA POMPAV. Anda bisa bertanya:\n{samples}",
                        "small_talk": f"🤖 Saya asisten QA POMPAV. Contoh pertanyaan:\n{samples}"
                    }
                    return responses[q_type], (0, 0, 0)

                # Cari dokumen yang relevan
                docs = self.vector_store.similarity_search(core_q if core_q else question, k=3)
                
                # Handle follow-up questions
                if (q_type == "follow_up" and self.memory.history 
                    and ("iya" not in question.lower() and "tolong" not in question.lower())):
                    
                    if not docs and self.last_topic:
                        # Coba cari berdasarkan topik terakhir
                        docs = self.vector_store.similarity_search(self.last_topic, k=3)
                    
                    if docs:
                        prompt_template = """
                        [KONTEKS SEBELUMNYA]
                        Pertanyaan: {last_question}
                        Jawaban: {last_answer}

                        [DOKUMEN TERKAIT]
                        {documents}

                        [PERTANYAAN LANJUTAN]
                        {question}

                        Berikan jawaban yang lebih mendalam berdasarkan konteks sebelumnya dan dokumen terkait.
                        """
                        
                        prompt = PromptTemplate(
                            template=prompt_template,
                            input_variables=["last_question", "last_answer", "documents", "question"]
                        )
                        
                        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
                        
                        response = llm_chain.run(
                            last_question=self.memory.history[-1]["question"],
                            last_answer=self.memory.history[-1]["answer"],
                            documents="\n\n".join([doc.page_content for doc in docs]),
                            question=question
                        )
                        
                        self.last_topic = core_q if core_q else question
                        self.last_answer = response
                        self.memory.add_interaction(question, response)
                        return response, (0.9, 0.9, 0.9)

                # Handle simple confirmation ("iya tolong")
                if "iya" in question.lower() or "tolong" in question.lower():
                    if self.last_topic:
                        docs = self.vector_store.similarity_search(self.last_topic, k=3)
                        if docs:
                            prompt_template = """
                            [DOKUMEN TERKAIT DENGAN TOPIK TERAKHIR]
                            {documents}

                            Berikan penjelasan lengkap tentang: {topic}
                            """
                            
                            prompt = PromptTemplate(
                                template=prompt_template,
                                input_variables=["documents", "topic"]
                            )
                            
                            llm_chain = LLMChain(llm=self.llm, prompt=prompt)
                            
                            response = llm_chain.run(
                                documents="\n\n".join([doc.page_content for doc in docs]),
                                topic=self.last_topic
                            )
                            
                            self.memory.add_interaction(question, response)
                            return response, (0.9, 0.9, 0.9)
                    return "Maaf, saya tidak yakin apa yang perlu dijelaskan. Bisakah Anda lebih spesifik?", (0, 0, 0)

                # Handle general questions
                if docs:
                    prompt_template = """
                    [DOKUMEN REFERENSI]
                    {context}

                    [PERTANYAAN]
                    {question}

                    Berikan jawaban singkat dan jelas berdasarkan dokumen di atas.
                    Jika pertanyaan tidak spesifik, berikan penjelasan umum.
                    """
                    
                    prompt = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )
                    
                    llm_chain = LLMChain(llm=self.llm, prompt=prompt)
                    
                    response = llm_chain.run(
                        context="\n\n".join([doc.page_content for doc in docs]),
                        question=core_q
                    )
                    
                    self.last_topic = core_q if core_q else question
                    self.last_answer = response
                    self.memory.add_interaction(question, response)
                    return response, (0.9, 0.9, 0.9)
                else:
                    return "❌ Informasi tidak ditemukan dalam dokumen. Mohon ajukan pertanyaan lain yang relevan.", (0, 0, 0)
            
            except Exception as e:
                traceback.print_exc()
                return f"⚠️ Error: {str(e)}", (0, 0, 0)

def main():
    try:
        if "OPENAI_API_KEY" not in os.environ:
            api_key = input("Masukkan OPENAI_API_KEY:")
            os.environ["OPENAI_API_KEY"] = api_key
    except Exception as e:
        print(f"⚠️ Error konfigurasi API: {str(e)}")
        return

    print("\n📤 Silakan masukkan path PDF dokumen POMPAV")
    pdf_path = input("Path file PDF: ").strip()

    print(f"\n🔍 Memproses dokumen: {pdf_path}...")
    processor = PDFProcessor()
    start = time.time()

    if not processor.initialize_vector_store(pdf_path):
        print("❌ Gagal memproses dokumen")
        return

    print(f"⏱️ Pemrosesan selesai dalam {time.time()-start:.2f} detik")

    qa_engine = QAEngine(processor.vector_store)

    print("\n" + "="*50)
    print("💬 SISTEM QA POMPAV (GPT-4) SIAP DIGUNAKAN")
    print("="*50)
    print("Ketik 'exit' atau 'keluar' untuk mengakhiri\n")

    while True:
        try:
            question = input("\n❓ Pertanyaan Anda: ").strip()

            if question.lower() in ['exit', 'keluar', 'quit', 'q']:
                break

            if not question:
                print("⚠️ Mohon masukkan pertanyaan")
                continue

            start_time = time.time()
            answer, _ = qa_engine.generate_response(question)

            print("\n💡 Jawaban:")
            print(answer)
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