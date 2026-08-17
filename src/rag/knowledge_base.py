"""Knowledge base management for the diabetes RAG pipeline."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _build_embeddings(
    embed_provider: str,
    ollama_base_url: Optional[str] = None,
    embed_model: Optional[str] = None,
    hf_model: Optional[str] = None,
    google_model: Optional[str] = None,
    max_seq_length: Optional[int] = None,
) -> Any:
    """Return a LangChain-compatible embedding object for the requested provider.

    Nama model embedding WAJIB sama antara ingest dan query. knowledge_base.py dan
    retriever.py sama-sama mengambilnya dari RagConfig agar tidak mungkin menyimpang:
    bila keduanya berbeda, retrieval merosot menjadi derau TANPA error apa pun.
    """
    from src.config import load_rag_config

    cfg = load_rag_config()
    ollama_base_url = ollama_base_url or cfg.ollama_base_url
    embed_model = embed_model or cfg.ollama_embed_model

    if embed_provider == "sentence-transformers":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        emb = HuggingFaceEmbeddings(
            model_name=hf_model or cfg.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        # WHAT : menimpa jendela token model embedding.
        # WHY  : max_seq_length 256 pada all-MiniLM-L6-v2 berasal dari
        #        sentence_bert_config.json — sebuah PILIHAN penulis model, bukan
        #        batas arsitektur. BERT di bawahnya ber-max_position_embeddings
        #        512, sehingga 257..512 adalah posisi yang bobotnya ADA.
        # WHEN : dipakai baik saat indexing maupun saat kueri. Menaikkannya hanya
        #        di salah satu sisi membuat dokumen dan kueri diwakili dengan dua
        #        aturan berbeda, dan retrieval merosot TANPA error apa pun.
        # HOW  : disetel pada objek SentenceTransformer di bawah LangChain.
        #        Default None = ikut bawaan model, sehingga perilaku lama utuh.
        batas = max_seq_length if max_seq_length is not None else cfg.embedding_max_seq_length
        if batas:
            klien = getattr(emb, "client", None)
            if klien is None or not hasattr(klien, "max_seq_length"):
                raise RuntimeError(
                    "max_seq_length diminta tetapi objek SentenceTransformer tidak "
                    "dapat dijangkau — menaikkannya diam-diam akan gagal tanpa jejak."
                )
            bawaan = klien.max_seq_length
            klien.max_seq_length = int(batas)
            logger.info("max_seq_length embedding: %s -> %s", bawaan, batas)
        return emb
    if embed_provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return _EmbeddingsBerlaju(GoogleGenerativeAIEmbeddings(
            model=google_model or cfg.google_embedding_model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        ))
    # Default: Ollama
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=embed_model, base_url=ollama_base_url)


# --------------------------------------------------------------------------
# Strategi pemecahan dokumen
#
# WHAT  : dua himpunan pemisah dan satu penakar panjang berbasis tokenizer.
# WHO   : dipakai MedicalKnowledgeBase.chunk_documents() di berkas ini, yang
#         menjadi satu-satunya jalur pemecahan korpus pedoman maupun manual_kb.
# WHERE : src/rag/knowledge_base.py; dipilih lewat argumen chunk_documents()
#         dan diteruskan scripts/reingest_kb.py dari baris perintah.
# WHEN   : hanya saat INDEXING (reingest), bukan saat kueri. Karena itu setiap
#         perubahan di sini menuntut indeks dibangun ulang; indeks lama tidak
#         ikut berubah dan akan diam-diam bercampur bila tidak dihapus dulu.
# WHY   : Gao dkk. (2023) §V.A.1 hal. 8 menyatakan pemecahan berukuran tetap
#         "leads to truncation within sentences". PEMISAH_KARAKTER di bawah
#         adalah persis kasus itu: setelah "\n" gagal ia langsung jatuh ke
#         spasi, sehingga batas potongan mendarat di sembarang kata.
#         PEMISAH_KALIMAT menyisipkan tanda akhir kalimat sebelum spasi,
#         sehingga spasi hanya dipakai bila satu kalimat memang lebih panjang
#         daripada satu potongan.
# HOW   : diserahkan ke RecursiveCharacterTextSplitter sebagai daftar
#         berprioritas. keep_separator="end" WAJIB — lihat catatan di bawah.
# --------------------------------------------------------------------------

# Perilaku lama (dipertahankan sebagai pembanding, bukan sebagai anjuran).
PEMISAH_KARAKTER = ["\n## ", "\n### ", "\n\n", "\n", " ", ""]

# URUTANNYA menentukan, dan urutan yang tampak wajar justru hampir tidak
# berpengaruh. Menyisipkan tanda akhir kalimat SESUDAH "\n" (yaitu
# [..., "\n\n", "\n", ". ", ...]) praktis tidak mengubah apa pun, karena "\n"
# selalu berhasil lebih dulu sehingga ". " tidak pernah sempat dipertimbangkan.
#
# Dasarnya: pada teks hasil ekstraksi PDF, "\n" TUNGGAL adalah pembungkusan
# baris — artefak tata letak halaman, bukan batas makna. "\n\n" (batas
# paragraf) dan judul tetap batas makna dan karena itu tetap didahulukan,
# tetapi akhir kalimat harus mengungguli pembungkusan baris.
#
# Diukur pada 531 halaman korpus, potongan 256 token:
#   kalimat SESUDAH "\n"  -> berakhir kalimat utuh 37,4% | bermula utuh 62,9%
#   kalimat SEBELUM "\n"  -> berakhir kalimat utuh 79,6% | bermula utuh 86,7%
#
# Varian ".\n" diperlukan terpisah: kalimat yang berakhir tepat di ujung baris
# menghasilkan ".\n" tanpa spasi, sehingga ". " saja tidak mengenainya.
PEMISAH_KALIMAT = [
    "\n## ", "\n### ", "\n\n",
    ". ", ".\n", "! ", "!\n", "? ", "?\n",
    "\n", "; ", " ", "",
]

# CATATAN YANG MENENTUKAN BENAR/SALAHNYA PERUBAHAN INI.
# Default RecursiveCharacterTextSplitter adalah keep_separator=True, yang
# menempelkan pemisah ke AWAL potongan berikutnya. Dengan pemisah ". " hasilnya
# menjadi potongan yang dibuka tanda titik ("​. Titrasi dilakukan ...") dan
# potongan sebelumnya kehilangan titiknya sendiri — yaitu memindahkan cacat,
# bukan memperbaikinya. Diverifikasi langsung pada versi terpasang:
#   keep_separator=True  -> ['... per kgBB', '. Titrasi tiap tiga hari', ...]
#   keep_separator="end" -> ['... per kgBB.', 'Titrasi tiap tiga hari.', ...]
KEEP_SEPARATOR_AKHIR = "end"

# Batas token model embedding produksi (all-MiniLM-L6-v2). Token ke-257 dan
# seterusnya DIBUANG tanpa peringatan apa pun. Diukur pada T5.1: 35,04% potongan
# produksi melewatinya dan 8,23% token korpus tidak pernah masuk vektor
# (results/eval_rag/distribusi_token.json).
BATAS_TOKEN_MINILM = 256


def _penakar_token(nama_model: str):
    """Kembalikan fungsi panjang yang menghitung TOKEN, bukan karakter.

    WHY: menakar panjang dengan len() berarti batas potongan diukur dengan satuan
    yang berbeda dari satuan yang dipakai model embedding. Selama dua satuan itu
    berbeda, TIDAK ADA nilai chunk_size karakter yang dapat menjamin potongan muat
    di jendela model — chunk_size 900 menghasilkan potongan 9 sampai 443 token.
    Menakar dengan tokenizer model itu sendiri membuat jaminannya bersifat
    konstruktif, bukan statistik.

    HOW: memakai tokenizer HuggingFace milik model embedding yang sama dengan yang
    dipakai saat kueri, sehingga tidak mungkin menyimpang.
    """
    from transformers import AutoTokenizer

    nama = nama_model if "/" in nama_model else f"sentence-transformers/{nama_model}"
    tok = AutoTokenizer.from_pretrained(nama)

    def panjang_token(teks: str) -> int:
        # add_special_tokens=True: [CLS] dan [SEP] ikut memakan jatah 256,
        # jadi mengabaikannya akan membuat potongan meleset dua token.
        return len(tok.encode(teks, add_special_tokens=True))

    return panjang_token


class _EmbeddingsBerlaju:
    """Bungkus embedding berbasis API dengan pembatas laju dan coba-ulang 429.

    WHAT  : pembatas laju sisi klien di depan penyedia embedding terkelola.
    WHO   : dipakai jalur ``google`` pada _build_embeddings(); jalur
            sentence-transformers lokal tidak memerlukannya.
    WHERE : src/rag/knowledge_base.py, membungkus GoogleGenerativeAIEmbeddings.
    WHEN  : setiap pemanggilan embed_documents/embed_query, yakni saat indexing
            korpus dan saat setiap kueri dokter.
    WHY   : free-tier Gemini membatasi 100 permintaan embedding per menit per
            model. Mengindeks 2.061 potongan sekaligus MELAMPAUINYA dan gagal di
            tengah jalan dengan 429, meninggalkan indeks separuh terisi — keadaan
            yang lebih berbahaya daripada gagal total karena tampak berhasil.
            Terverifikasi: percobaan pertama T10 gagal persis begitu.
    HOW   : permintaan dipecah menjadi kelompok kecil, dijeda agar lajunya di
            bawah batas, dan 429 dicoba ulang dengan mundur bertahap. Angka
            bawaan disetel konservatif (90/menit) karena batasnya menghitung
            KONTEN, bukan permintaan batch.

    Sengaja bukan turunan kelas LangChain: antarmuka yang dipakai Chroma hanya
    embed_documents dan embed_query, dan membungkus jauh lebih tahan terhadap
    perubahan versi pustaka daripada mewarisi.
    """

    def __init__(self, inner: Any, per_menit: int = 90, ukuran_kelompok: int = 30,
                 maks_coba: int = 5):
        self._inner = inner
        self._per_menit = max(per_menit, 1)
        self._ukuran = max(ukuran_kelompok, 1)
        self._maks_coba = max(maks_coba, 1)

    def __getattr__(self, nama: str) -> Any:
        # Atribut lain (mis. .model) diteruskan apa adanya.
        return getattr(self._inner, nama)

    def _coba_ulang(self, fungsi, *a):
        import time
        for percobaan in range(self._maks_coba):
            try:
                return fungsi(*a)
            except Exception as exc:
                if "429" not in str(exc) and "quota" not in str(exc).lower():
                    raise
                if percobaan == self._maks_coba - 1:
                    raise
                jeda = 30 * (percobaan + 1)
                logger.warning("Kuota embedding tercapai; menunggu %d dtk (percobaan %d/%d)",
                               jeda, percobaan + 1, self._maks_coba)
                time.sleep(jeda)
        raise RuntimeError("tidak tercapai")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import time
        hasil: List[List[float]] = []
        jeda_per_kelompok = 60.0 * self._ukuran / self._per_menit
        total = (len(texts) + self._ukuran - 1) // max(self._ukuran, 1)
        for i in range(0, len(texts), self._ukuran):
            t0 = time.time()
            hasil.extend(self._coba_ulang(self._inner.embed_documents,
                                          texts[i:i + self._ukuran]))
            if i // self._ukuran % 10 == 0:
                logger.info("Embedding kelompok %d/%d", i // self._ukuran + 1, total)
            # Jeda hanya bila masih ada sisa; menjeda sesudah kelompok terakhir
            # hanya memperlambat tanpa manfaat.
            if i + self._ukuran < len(texts):
                time.sleep(max(0.0, jeda_per_kelompok - (time.time() - t0)))
        return hasil

    def embed_query(self, text: str) -> List[float]:
        # Kueri juga dihitung kuota. Evaluasi menembakkan ratusan kueri berturut-turut
        # sehingga tanpa jeda ia menabrak batas yang sama seperti indexing.
        import time
        minimal = 60.0 / self._per_menit
        sejak = time.time() - getattr(self, "_terakhir", 0.0)
        if sejak < minimal:
            time.sleep(minimal - sejak)
        try:
            return self._coba_ulang(self._inner.embed_query, text)
        finally:
            self._terakhir = time.time()


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Ubah metadata agar kompatibel ChromaDB (hanya skalar str/int/float/bool).

    list/tuple → gabung jadi string; dict/lainnya → str().
    Kunci bernilai None DIBUANG: ChromaDB menolak nilai None saat upsert, jadi
    menyimpannya hanya menunda kegagalan ke titik yang lebih sulit didiagnosis.
    Metadata halaman memakai pasangan sentinel (halaman_cetak=0 +
    halaman_cetak_valid=False), bukan None — lihat scripts/reingest_kb.py.
    """
    clean: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = ", ".join(str(v) for v in value)
        else:
            clean[key] = str(value)
    return clean


class MedicalKnowledgeBase:
    """Load, chunk, and persist medical knowledge for retrieval."""

    def __init__(
        self,
        kb_dir: Optional[str] = None,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embed_provider: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        embed_model: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        from src.config import load_rag_config

        cfg = config or load_rag_config()
        self.cfg = cfg

        self.kb_dir = Path(kb_dir or cfg.knowledge_base_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)

        self.persist_dir = Path(persist_dir or cfg.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name or cfg.collection_name
        self.embed_provider = embed_provider or cfg.embedding_provider
        self.ollama_base_url = ollama_base_url or cfg.ollama_base_url
        self.embed_model = embed_model or cfg.ollama_embed_model

        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []

    def muat_potongan_cadangan(
        self, file_name: str = "fallback_chunks.json"
    ) -> List[Dict[str, Any]]:
        """Muat potongan cadangan KNF-04 yang diekspor dari korpus pedoman.

        MENGGANTIKAN ``load_manual_kb`` (dicabut 17 Agustus 2026). Metode lama membaca
        ``manual_kb.json``, yakni prosa yang disusun sendiri oleh peneliti dan TIDAK
        memiliki nomor halaman sumber, sehingga potongannya tampil sebagai "Hal. tidak
        tercatat" pada antarmuka. Selama potongan itu berada di dalam indeks, klaim KNF-08
        bahwa tiap potongan tertelusur sampai halamannya tidak benar.

        Berkas penggantinya diekspor ``scripts/reingest_kb.py`` DARI korpus pedoman, dan
        tiap entrinya sudah membawa metadata sitasi yang lengkap. Ia dipakai hanya ketika
        ChromaDB tidak dapat dibuka, sebagai cadangan berjangkauan sempit.

        Mengembalikan daftar kosong bila berkasnya belum ada; pemanggilnya yang menentukan
        apakah itu keadaan yang dapat ditoleransi.
        """
        # HANYA kb_dir yang dibaca, tanpa jatuh ke lintasan bawaan repositori.
        #
        # Metode lama menyimpan cadangan tersembunyi ke "data/knowledge_base" bila berkas
        # tidak ada di kb_dir. Akibatnya kb_dir yang diberikan pemanggil DIABAIKAN diam-diam,
        # dan pemanggil memperoleh potongan dari korpus produksi tanpa menyadarinya —
        # persis jenis perilaku senyap yang berulang kali menyesatkan pada penelitian ini.
        jalur = self.kb_dir / file_name
        if not jalur.exists():
            logger.warning(
                "Potongan cadangan tidak ditemukan: %s. Jalankan "
                "scripts/reingest_kb.py untuk membangunnya.", jalur)
            return []

        with jalur.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        docs = [p for p in payload if str(p.get("text", "")).strip()]
        self.documents = docs
        logger.info("Memuat %d potongan cadangan dari %s", len(docs), jalur)
        return docs

    def load_documents(self, file_pattern: str = "*.txt") -> None:
        """Compatibility helper to load text files from knowledge base directory."""
        files = list(self.kb_dir.glob(file_pattern))
        loaded_docs: List[Dict[str, Any]] = []

        for idx, path in enumerate(files):
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            metadata = self._build_metadata(
                source=path.name,
                topic=path.stem,
                doc_id=f"txt_{path.stem}",
                index=idx,
            )
            loaded_docs.append(
                {
                    "text": text,
                    "source": path.name,
                    "topic": path.stem,
                    "metadata": metadata,
                }
            )

        self.documents = loaded_docs
        logger.info("Loaded %d text documents from %s", len(loaded_docs), self.kb_dir)

    def chunk_documents(
        self,
        documents: Optional[List[Dict[str, Any]]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        pemisah_kalimat: Optional[bool] = None,
        satuan_panjang: str = "karakter",
    ) -> List[Dict[str, Any]]:
        """Pecah dokumen menjadi potongan siap-indeks.

        pemisah_kalimat
            None (default) mengambil nilai dari rag.pemisah_kalimat pada config,
            sehingga produksi dan percobaan memakai aturan yang sama.
            False mempertahankan perilaku lama: setelah baris baru,
            pemisah berikutnya adalah spasi, sehingga potongan berhenti di
            sembarang kata. True memakai PEMISAH_KALIMAT.
        satuan_panjang
            "karakter" (default) menakar chunk_size dengan len(). "token"
            menakarnya dengan tokenizer model embedding, sehingga chunk_size
            berarti JUMLAH TOKEN dan pemotongan senyap 256 token menjadi
            mustahil secara konstruktif.

        Keduanya sengaja default ke perilaku lama: mengubah default akan
        membuat indeks yang sudah dilaporkan angkanya berubah diam-diam.
        Pemilihan dilakukan eksplisit oleh scripts/reingest_kb.py.
        """
        chunk_size = chunk_size if chunk_size is not None else self.cfg.chunk_size
        chunk_overlap = chunk_overlap if chunk_overlap is not None else self.cfg.chunk_overlap
        if pemisah_kalimat is None:
            pemisah_kalimat = getattr(self.cfg, "pemisah_kalimat", False)
        docs = documents if documents is not None else self.documents
        if not docs:
            logger.warning("No documents available to chunk")
            self.chunks = []
            return []

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            opsi: Dict[str, Any] = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "separators": PEMISAH_KALIMAT if pemisah_kalimat else PEMISAH_KARAKTER,
            }
            if pemisah_kalimat:
                # Hanya bermakna bila pemisahnya tanda baca; lihat KEEP_SEPARATOR_AKHIR.
                opsi["keep_separator"] = KEEP_SEPARATOR_AKHIR
            if satuan_panjang == "token":
                opsi["length_function"] = _penakar_token(self.cfg.embedding_model)

            splitter = RecursiveCharacterTextSplitter(**opsi)
            logger.info(
                "Chunker: satuan=%s ukuran=%d overlap=%d pemisah_kalimat=%s",
                satuan_panjang, chunk_size, chunk_overlap, pemisah_kalimat,
            )

            chunk_rows: List[Dict[str, Any]] = []
            for doc_idx, doc in enumerate(docs):
                pieces = splitter.split_text(doc["text"])
                for part_idx, part in enumerate(pieces):
                    metadata = dict(doc.get("metadata", {}))
                    metadata.update(
                        {
                            "chunk_index": part_idx,
                            "chunk_id": f"{metadata.get('doc_id', f'doc_{doc_idx}')}_ch_{part_idx:03d}",
                        }
                    )
                    chunk_rows.append(
                        {
                            "text": part,
                            "source": doc.get("source", "manual_kb"),
                            "topic": doc.get("topic", "general"),
                            "metadata": metadata,
                        }
                    )

            self.chunks = chunk_rows
            logger.info("Chunked %d docs into %d chunks", len(docs), len(chunk_rows))
            return chunk_rows

        except Exception as exc:
            logger.warning("LangChain splitter unavailable, fallback splitter active: %s", exc)
            chunk_rows = self._fallback_chunk_documents(docs, chunk_size=chunk_size, overlap=chunk_overlap)
            self.chunks = chunk_rows
            return chunk_rows

    def _fallback_chunk_documents(
        self,
        docs: List[Dict[str, Any]],
        chunk_size: int,
        overlap: int,
    ) -> List[Dict[str, Any]]:
        """Simple token-window chunking fallback when text splitter is unavailable."""
        chunks: List[Dict[str, Any]] = []
        step = max(chunk_size - overlap, 1)

        for doc_idx, doc in enumerate(docs):
            words = doc["text"].split()
            for start in range(0, len(words), step):
                part = " ".join(words[start : start + chunk_size]).strip()
                if not part:
                    continue
                metadata = dict(doc.get("metadata", {}))
                chunk_number = len(chunks)
                metadata.update(
                    {
                        "chunk_index": chunk_number,
                        "chunk_id": f"{metadata.get('doc_id', f'doc_{doc_idx}')}_ch_{chunk_number:03d}",
                    }
                )
                chunks.append(
                    {
                        "text": part,
                        "source": doc.get("source", "manual_kb"),
                        "topic": doc.get("topic", "general"),
                        "metadata": metadata,
                    }
                )

        logger.info("Fallback chunker created %d chunks", len(chunks))
        return chunks

    def _build_metadata(self, source: str, topic: str, doc_id: str, index: int) -> Dict[str, Any]:
        """Build normalized metadata fields for each document/chunk."""
        return {
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_ch_000",
            "domain": "manual",
            "subdomain": topic.lower().replace(" ", "_"),
            "sumber": "Manual KB",
            "judul": topic,
            "tahun": 2024,
            "versi": "v1",
            "url_sumber": "",
            "halaman": "N/A",
            "jenis_dm": ["dm_tipe2"],
            "setting": ["fktp", "fkrtl"],
            "sasaran": ["dokter_umum", "sppd"],
                "populasi_khusus": ["umum"],
            "tipe_konten": "panduan_klinis",
            "bahasa": "id",
            "level_bukti": "guideline",
            "topik_terkait": [topic],
            "perlu_update_sebelum": "2026-12",
            "status": "aktif",
            "source": source,
            "created_at": datetime.utcnow().isoformat(),
            "index": index,
        }

    def save_to_chroma(self, chunks: Optional[List[Dict[str, Any]]] = None, reset_collection: bool = False) -> bool:
        """Persist chunks into ChromaDB using the configured embedding provider."""
        chunk_rows = chunks if chunks is not None else self.chunks
        if not chunk_rows:
            logger.warning("No chunks available for Chroma ingestion")
            return False

        try:
            from langchain_chroma import Chroma
            from langchain_core.documents import Document
        except Exception as exc:
            logger.error("Chroma dependencies unavailable: %s", exc)
            return False

        try:
            embeddings = _build_embeddings(
                self.embed_provider, self.ollama_base_url, self.embed_model,
                hf_model=self.cfg.embedding_model,
                google_model=self.cfg.google_embedding_model,
            )
        except Exception as exc:
            logger.error("Embedding initialisation failed (%s): %s", self.embed_provider, exc)
            return False

        # hnsw:space=cosine — default Chroma adalah l2, yang membuat skor relevansi
        # LangChain (1 - d/sqrt(2)) bisa negatif dan memicu peringatan "di luar [0,1]".
        # Embedding sudah dinormalisasi (normalize_embeddings=True), sehingga cosine
        # memberi skor [0,1] yang dapat ditampilkan langsung ke dokter.
        collection_metadata = {"hnsw:space": "cosine"}

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_dir),
            collection_metadata=collection_metadata,
        )

        if reset_collection:
            try:
                vector_store.delete_collection()
                vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=embeddings,
                    persist_directory=str(self.persist_dir),
                    collection_metadata=collection_metadata,
                )
            except Exception as exc:
                logger.warning("Could not reset existing collection: %s", exc)

        documents: List[Document] = []
        for row in chunk_rows:
            raw_meta = {
                **dict(row.get("metadata", {})),
                "source": row.get("source", "manual_kb"),
                "topic": row.get("topic", "general"),
            }
            # ChromaDB hanya menerima metadata skalar (str/int/float/bool/None);
            # list/dict di-flatten ke string agar tidak ditolak saat upsert.
            documents.append(
                Document(page_content=row["text"], metadata=_sanitize_metadata(raw_meta))
            )

        # ChromaDB menolak batch di atas batas internalnya (5.461 pada versi ini) dengan
        # InternalError, BUKAN dengan pesan yang menyarankan pemecahan. Korpus produksi
        # sekarang 2.061 chunk sehingga belum pernah menyentuhnya, tetapi ukuran potongan
        # yang lebih kecil langsung melewatinya: chunk_size=300 menghasilkan 5.896 chunk
        # dan seluruh ingest gagal. Memecah di sini membuat batasnya tidak lagi menjadi
        # batas korpus.
        BATCH = 4000
        for mulai in range(0, len(documents), BATCH):
            vector_store.add_documents(documents[mulai:mulai + BATCH])
        logger.info("Saved %d chunks to ChromaDB at %s (%d batch)",
                    len(documents), self.persist_dir,
                    (len(documents) + BATCH - 1) // max(BATCH, 1))
        return True

    def save_chunks(self, output_path: Optional[Path] = None) -> None:
        """Save prepared chunks into JSON for traceability and tests."""
        target = output_path or (self.kb_dir / "chunks.json")
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.chunks, handle, ensure_ascii=False, indent=2)

    def load_chunks(self, filepath: Optional[Path] = None) -> None:
        """Load precomputed chunks from JSON file."""
        source_path = filepath or (self.kb_dir / "chunks.json")
        with source_path.open("r", encoding="utf-8") as handle:
            self.chunks = json.load(handle)

    def process_all_documents(self, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> None:
        """Compatibility wrapper used by existing code paths."""
        self.chunk_documents(documents=self.documents, chunk_size=chunk_size, chunk_overlap=overlap)

    # ``create_manual_kb`` DICABUT pada 17 Agustus 2026.
    #
    # Metode itu menuliskan dua potongan contoh berisi definisi hiperglikemia dan
    # hipoglikemia yang disusun sendiri, lalu menyimpannya sebagai manual_kb.json. Isinya
    # tidak berasal dari pedoman terbitan resmi dan tidak membawa nomor halaman, sehingga
    # keberadaannya di dalam indeks membuat klaim keterlacakan sitasi tidak benar.
    #
    # Penggantinya bukan metode lain, melainkan jalur yang berbeda sama sekali:
    # ``scripts/reingest_kb.py`` membangun korpus dari PDF pedoman per halaman beserta
    # manifest, lalu mengekspor sebagian potongannya sebagai cadangan penelusuran yang
    # dibaca ``muat_potongan_cadangan``.
