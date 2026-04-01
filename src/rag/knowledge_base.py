"""
Knowledge Base Management for RAG
Handles extraction and chunking of medical documents
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
    logging.warning("PyPDF2 not installed, PDF extraction will not work")

logger = logging.getLogger(__name__)


class MedicalKnowledgeBase:
    """
    Manage medical knowledge base for RAG retrieval
    """
    
    def __init__(self, kb_dir: str = "data/knowledge_base"):
        """
        Initialize Knowledge Base
        
        Args:
            kb_dir: Directory containing medical documents
        """
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        
        self.documents = []
        self.chunks = []
        self.metadata = {}
        
        logger.info(f"Knowledge Base initialized at {kb_dir}")
    
    def extract_pdf(self, pdf_path: Path, output_path: Optional[Path] = None) -> str:
        """
        Extract text from PDF
        
        Args:
            pdf_path: Path to PDF file
            output_path: Optional path to save extracted text
            
        Returns:
            text: Extracted text content
        """
        if PyPDF2 is None:
            raise ImportError("PyPDF2 required for PDF extraction. Install: pip install PyPDF2")
        
        logger.info(f"Extracting text from {pdf_path}...")
        
        with open(pdf_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            
            full_text = ""
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                # Clean text
                text = self._clean_text(text)
                
                full_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            logger.info(f"Extracted text saved to {output_path}")
        
        logger.info(f"Extracted {len(full_text)} characters from {pdf_path.name}")
        
        return full_text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw text
            
        Returns:
            cleaned: Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (standalone numbers)
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 500, 
                   overlap: int = 50) -> List[Dict]:
        """
        Split text into overlapping chunks for better context retrieval
        
        Args:
            text: Input text
            chunk_size: Number of words per chunk
            overlap: Number of words to overlap between chunks
            
        Returns:
            chunks: List of chunk dictionaries
        """
        logger.info(f"Chunking text (size={chunk_size}, overlap={overlap})...")
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'start_index': i,
                'end_index': i + len(chunk_words),
                'chunk_id': len(chunks)
            })
        
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def load_documents(self, file_pattern: str = "*.txt") -> None:
        """
        Load all documents from knowledge base directory
        
        Args:
            file_pattern: Glob pattern for files to load
        """
        files = list(self.kb_dir.glob(file_pattern))
        
        logger.info(f"Loading {len(files)} documents...")
        
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            self.documents.append({
                'filename': filepath.name,
                'text': text,
                'source': str(filepath)
            })
        
        logger.info(f"Loaded {len(self.documents)} documents")
    
    def process_all_documents(self, chunk_size: int = 500, 
                             overlap: int = 50) -> None:
        """
        Process all loaded documents into chunks
        
        Args:
            chunk_size: Chunk size in words
            overlap: Overlap in words
        """
        if not self.documents:
            logger.warning("No documents loaded. Call load_documents() first.")
            return
        
        all_chunks = []
        
        for doc in self.documents:
            chunks = self.chunk_text(doc['text'], chunk_size, overlap)
            
            # Add metadata
            for chunk in chunks:
                chunk['source'] = doc['filename']
                all_chunks.append(chunk)
        
        self.chunks = all_chunks
        
        logger.info(f"Processed {len(self.documents)} documents into {len(self.chunks)} chunks")
    
    def save_chunks(self, output_path: Optional[Path] = None) -> None:
        """
        Save chunks to JSON
        
        Args:
            output_path: Path to save chunks
        """
        if output_path is None:
            output_path = self.kb_dir / 'chunks.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.chunks)} chunks to {output_path}")
    
    def load_chunks(self, filepath: Optional[Path] = None) -> None:
        """
        Load pre-processed chunks from JSON
        
        Args:
            filepath: Path to chunks JSON
        """
        if filepath is None:
            filepath = self.kb_dir / 'chunks.json'
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        logger.info(f"Loaded {len(self.chunks)} chunks from {filepath}")
    
    def create_manual_kb(self) -> None:
        """
        Create minimal manual knowledge base (Plan C fallback)
        """
        logger.info("Creating manual knowledge base...")
        
        manual_kb = [
            {
                'topic': 'Hiperglikemia',
                'text': """
                Hiperglikemia adalah kondisi di mana kadar gula darah melebihi 180 mg/dL.
                
                Penyebab utama:
                - Asupan karbohidrat berlebihan
                - Dosis insulin tidak mencukupi
                - Tingkat stress yang tinggi
                - Kurangnya aktivitas fisik
                - Infeksi atau sakit
                
                Gejala:
                - Sering buang air kecil
                - Rasa haus berlebihan
                - Pandangan kabur
                - Kelelahan
                - Sakit kepala
                
                Penanganan:
                1. Perbanyak minum air putih (minimal 2-3 gelas)
                2. Lakukan aktivitas fisik ringan 15-30 menit
                3. Hindari makanan tinggi gula
                4. Periksa keton jika glukosa >250 mg/dL
                5. Konsultasi dokter jika bertahan >2 jam atau >300 mg/dL
                
                Referensi: PERKENI 2021, ADA Standards of Care 2023
                """
            },
            {
                'topic': 'Hipoglikemia',
                'text': """
                Hipoglikemia adalah kondisi gula darah di bawah 70 mg/dL. Ini adalah kondisi DARURAT.
                
                Penyebab:
                - Dosis insulin berlebihan
                - Makan terlambat atau melewatkan makan
                - Aktivitas fisik berlebihan tanpa penyesuaian insulin
                - Konsumsi alkohol
                
                Gejala:
                - Gemetar, berkeringat
                - Jantung berdebar
                - Pusing, kebingungan
                - Lapar ekstrem
                - Mudah marah
                - Penglihatan kabur
                
                Aturan 15-15 (Penanganan SEGERA):
                1. Konsumsi 15 gram karbohidrat cepat:
                   - 3-4 tablet glukosa
                   - 120 ml jus buah
                   - 1 sendok makan madu
                   - 5-6 permen keras
                2. Tunggu 15 menit
                3. Cek ulang gula darah
                4. Jika masih <70 mg/dL, ulangi langkah 1-3
                5. Setelah normal, makan camilan dengan protein
                
                PERINGATAN: Jika kehilangan kesadaran, SEGERA hubungi 119 atau bawa ke UGD.
                
                Referensi: ADA Hypoglycemia Guidelines, PERKENI 2021
                """
            },
            {
                'topic': 'Manajemen Stress',
                'text': """
                Stress dapat meningkatkan kadar gula darah melalui pelepasan hormon kortisol dan adrenalin.
                
                Dampak stress terhadap diabetes:
                - Meningkatkan resistensi insulin
                - Memicu pelepasan glukosa dari hati
                - Mengganggu pola makan dan tidur
                - Menurunkan motivasi untuk kontrol diabetes
                
                Teknik manajemen stress:
                1. Pernapasan dalam (4-7-8):
                   - Tarik napas 4 detik
                   - Tahan 7 detik
                   - Buang 8 detik
                   - Ulangi 5-10 kali
                
                2. Meditasi mindfulness 10-15 menit
                3. Aktivitas fisik teratur
                4. Tidur cukup 7-8 jam
                5. Dukungan sosial dari keluarga/support group
                
                Jika stress berkepanjangan, konsultasi dengan psikolog atau konselor diabetes.
                
                Referensi: ADA Standards of Care - Psychosocial Care
                """
            },
            {
                'topic': 'Aktivitas Fisik',
                'text': """
                Aktivitas fisik adalah komponen penting manajemen diabetes tipe 2.
                
                Manfaat:
                - Meningkatkan sensitivitas insulin
                - Menurunkan kadar gula darah
                - Membantu kontrol berat badan
                - Mengurangi risiko komplikasi kardiovaskular
                - Meningkatkan mood dan mengurangi stress
                
                Rekomendasi ADA:
                - Minimal 150 menit/minggu aktivitas aerobik intensitas sedang
                - Latihan kekuatan 2-3x per minggu
                - Hindari duduk >30 menit terus-menerus
                
                Jenis aktivitas:
                1. Aerobik: jalan cepat, jogging, bersepeda, berenang
                2. Kekuatan: angkat beban, resistance band, yoga
                3. Fleksibilitas: stretching, tai chi
                
                PERINGATAN untuk pengguna insulin:
                - Cek gula darah sebelum dan sesudah olahraga
                - Jika <100 mg/dL sebelum olahraga, konsumsi camilan 15g karbohidrat
                - Bawa sumber glukosa cepat saat berolahraga
                - Kurangi dosis insulin jika planning olahraga intens
                
                Referensi: ADA Physical Activity Guidelines, PERKENI 2021
                """
            },
            {
                'topic': 'Karbohidrat dan Meal Planning',
                'text': """
                Manajemen asupan karbohidrat adalah kunci kontrol gula darah.
                
                Prinsip Carbohydrate Counting:
                - 1 unit insulin umumnya untuk 10-15g karbohidrat (sesuaikan dengan rasio individu)
                - Konsistensi jumlah karbohidrat antar hari membantu prediksi
                
                Sumber karbohidrat sehat:
                1. Karbohidrat kompleks (diutamakan):
                   - Nasi merah, oat, quinoa
                   - Roti gandum utuh
                   - Kentang, ubi
                   - Kacang-kacangan
                
                2. Sayuran non-starch (bebas):
                   - Bayam, kangkung, sawi
                   - Brokoli, kembang kol
                   - Tomat, mentimun
                
                3. Buah (porsi terkontrol):
                   - Apel, jeruk, pir (1 buah sedang = 15g)
                   - Pisang kecil = 15g
                   - Hindari jus buah (kehilangan serat)
                
                Tips praktis:
                - Gunakan metode piring: 1/2 sayur, 1/4 protein, 1/4 karbohidrat
                - Baca label nutrisi untuk hitung karbohidrat
                - Makan teratur, hindari melewatkan waktu makan
                - Kombinasikan karbohidrat dengan protein/lemak sehat untuk memperlambat absorpsi
                
                Referensi: PERKENI Nutrition Therapy, ADA Nutrition Guidelines
                """
            },
            {
                'topic': 'Target Kontrol Glikemik',
                'text': """
                Target kontrol gula darah untuk dewasa diabetes tipe 2 (non-hamil):
                
                Gula Darah Puasa (GDP):
                - Target: 80-130 mg/dL
                - Optimal: 90-110 mg/dL
                
                Gula Darah 2 jam Setelah Makan (GDPP):
                - Target: <180 mg/dL
                - Optimal: <140 mg/dL
                
                HbA1c (rata-rata 3 bulan):
                - Target umum: <7%
                - Target ketat (jika aman): <6.5%
                - Target longgar (lansia, komorbid): <8%
                
                Time in Range (jika pakai CGM):
                - >70% waktu dalam range 70-180 mg/dL
                - <4% waktu di bawah 70 mg/dL
                - <25% waktu di atas 180 mg/dL
                
                Kapan konsultasi dokter:
                - GDP sering >180 mg/dL
                - Hipoglikemia >2x per minggu
                - HbA1c tidak mencapai target setelah 3 bulan
                - Gejala komplikasi (kebas, luka tidak sembuh, pandangan kabur)
                
                Referensi: ADA Standards of Care 2023, PERKENI 2021
                """
            },
            {
                'topic': 'Monitoring dan Self-Care',
                'text': """
                Pemantauan mandiri adalah kunci sukses manajemen diabetes.
                
                Frekuensi Monitoring Gula Darah:
                1. Pengguna insulin:
                   - Minimal 4x sehari (sebelum makan dan sebelum tidur)
                   - Tambahan saat sakit, olahraga, atau gejala hipo/hiper
                
                2. Tanpa insulin:
                   - Minimal 2x seminggu (bervariasi waktu)
                   - Lebih sering saat penyesuaian obat
                
                Cara Monitoring yang Benar:
                - Cuci tangan dengan sabun dan air (JANGAN pakai alkohol swab di jari)
                - Tusuk sisi ujung jari (lebih sedikit saraf)
                - Tetes darah pertama cukup, jangan diperas berlebihan
                - Catat hasil beserta waktu, makanan, dan aktivitas
                
                Pencatatan (Logbook):
                - Tanggal dan waktu
                - Hasil gula darah
                - Makanan dan jumlah karbohidrat
                - Dosis dan waktu insulin/obat
                - Aktivitas fisik
                - Tingkat stress atau perasaan
                - Gejala tidak biasa
                
                Perawatan Kaki (penting untuk cegah komplikasi):
                - Cek kaki setiap hari
                - Jaga kebersihan dan kelembaban
                - Gunakan alas kaki yang nyaman
                - Segera konsultasi jika ada luka
                
                Referensi: ADA Self-Management Education, PERKENI 2021
                """
            }
        ]
        
        # Convert to chunks format
        self.chunks = []
        for idx, item in enumerate(manual_kb):
            self.chunks.append({
                'text': item['text'],
                'source': 'manual_kb',
                'topic': item['topic'],
                'chunk_id': idx
            })
        
        logger.info(f"Created manual KB with {len(self.chunks)} entries")
        
        # Save
        self.save_chunks(self.kb_dir / 'manual_kb.json')