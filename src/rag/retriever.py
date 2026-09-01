"""Retrieval layer for diabetes RAG using Chroma MMR with fallback keyword scoring."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Pembangun embedding dipakai bersama knowledge_base.py — memakai SATU implementasi
# menjamin model embedding kueri dan dokumen tidak mungkin menyimpang. Bila keduanya
# berbeda, retrieval merosot menjadi derau tanpa error apa pun.
from .knowledge_base import _build_embeddings  # noqa: E402


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class SimpleKeywordRetriever:
    """Fallback retriever used when vector infrastructure is unavailable."""

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if top_k is None:
            from src.config import load_rag_config
            top_k = load_rag_config().top_k
        query_tokens = Counter(_tokenize(query))
        results: List[Dict[str, Any]] = []

        for chunk in self.chunks:
            meta = dict(chunk.get("metadata", {}))
            if metadata_filter and not _metadata_matches(meta, metadata_filter):
                continue

            text = chunk.get("text", "")
            chunk_tokens = Counter(_tokenize(text))
            overlap = sum(min(query_tokens[token], chunk_tokens[token]) for token in query_tokens)
            similarity = overlap / max(len(query_tokens), 1)
            results.append(
                {
                    "rank": 0,
                    "text": text,
                    "source": chunk.get("source", "manual_kb"),
                    "similarity": float(similarity),
                    "metadata": meta,
                }
            )

        results.sort(key=lambda item: item["similarity"], reverse=True)
        for idx, row in enumerate(results[:top_k], start=1):
            row["rank"] = idx
        return results[:top_k]


class MMRRetriever:
    """ChromaDB retriever using MMR search strategy with configurable embeddings.

    Supports three embedding providers:
    - ``"sentence-transformers"`` — CPU-only, no server needed (default, recommended)
    - ``"google"``               — Google Generative AI embeddings (requires GOOGLE_API_KEY)
    - ``"ollama"``               — local Ollama server (legacy)
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embed_provider: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        embed_model: Optional[str] = None,
        top_k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        config: Optional[Any] = None,
        retrieval_mode: Optional[str] = None,
    ):
        from src.config import load_rag_config

        cfg = config or load_rag_config()
        self.cfg = cfg

        self.persist_dir = persist_dir or cfg.persist_dir
        self.collection_name = collection_name or cfg.collection_name
        self.embed_provider = embed_provider or cfg.embedding_provider
        self.ollama_base_url = ollama_base_url or cfg.ollama_base_url
        self.embed_model = embed_model or cfg.ollama_embed_model
        self.top_k = top_k if top_k is not None else cfg.top_k
        self.fetch_k = fetch_k if fetch_k is not None else cfg.fetch_k
        self.lambda_mult = lambda_mult if lambda_mult is not None else cfg.lambda_mult

        # T13/T14 — cara menelusur. Lihat catatan panjang pada config.yaml.
        self.retrieval_mode = (retrieval_mode
                               or getattr(cfg, "retrieval_mode", "vektor")).lower()
        self.rrf_pool = int(getattr(cfg, "rrf_pool", 50))
        self.rrf_k = int(getattr(cfg, "rrf_k", 60))

        self.reranker_enabled = bool(
            getattr(getattr(cfg, "reranker", None), "enabled", False)
        )
        self.reranker_backend = str(getattr(
            getattr(cfg, "reranker", None),
            "backend",
            "cross_encoder",
        )).lower()
        self.reranker_model = getattr(
            getattr(cfg, "reranker", None),
            "model",
            "BAAI/bge-reranker-v2-m3",
        )
        self.reranker_candidate_k = int(
            getattr(getattr(cfg, "reranker", None), "candidate_k", 50)
        )
        self.reranker_max_length = int(
            getattr(getattr(cfg, "reranker", None), "max_length", 512)
        )

        self._reranker = None
        self._reranker_error = None

        self._vector_store = None
        self._embeddings = None
        self._bm25 = None
        self._bm25_teks: List[str] = []
        self._bm25_metas: List[Optional[Dict[str, Any]]] = []
        self._bm25_peta: Dict[str, int] = {}
        self._bm25_error: Optional[str] = None
        self.mode_diminta = self.retrieval_mode
        self._init_error: Optional[str] = None

        try:
            # Mode "bm25" murni LEKSIKAL: penelusuran tidak pernah menyentuh ruang
            # vektor. _bangun_bm25 dan _hasil_dari_indeks hanya membaca `documents`
            # dan `metadatas` dari sqlite koleksi. Membangun fungsi embedding di sini
            # berarti memuat torch dan model kalimat (~1 GB) yang TIDAK PERNAH dipakai
            # menelusur — beban mati yang menghabiskan memori instans kecil.
            #
            # Jalur vektor dan hibrida tetap membutuhkannya dan tidak berubah.
            if self.retrieval_mode == "bm25":
                self._bangun_bm25()

                if self._bm25 is None:
                    # _bangun_bm25 sudah menurunkan mode ke "vektor" secara eksplisit
                    # dan menyimpan sebabnya. Penurunan itu hanya bermakna bila ruang
                    # vektornya benar-benar ada, jadi bangun sekarang.
                    self._bangun_vector_store(cfg)
            else:
                self._bangun_vector_store(cfg)

                if self.retrieval_mode == "hibrida":
                    self._bangun_bm25()

            self._muat_reranker()

            logger.info(
                "MMRRetriever connected — collection=%s embed=%s mode=%s vektor=%s",
                self.collection_name,
                self.embed_provider,
                self.retrieval_mode,
                self._vector_store is not None,
            )
        except Exception as exc:
            self._init_error = str(exc)
            logger.warning(
                "MMRRetriever unavailable (embed=%s), keyword fallback required: %s",
                embed_provider,
                exc,
            )

    def _bangun_vector_store(self, cfg: Any) -> None:
        """Bangun fungsi embedding dan koleksi Chroma untuk penelusuran vektor.

        Dipisahkan dari ``__init__`` supaya jalur leksikal dapat melewatinya. Inilah
        satu-satunya tempat torch dan model kalimat dimuat.
        """
        from langchain_chroma import Chroma

        embeddings = _build_embeddings(
            self.embed_provider, self.ollama_base_url, self.embed_model,
            hf_model=cfg.embedding_model, google_model=cfg.google_embedding_model,
        )
        self._embeddings = embeddings

        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_dir,
        )

    def _muat_reranker(self) -> None:
        """Muat cross-encoder HANYA bila jalur yang aktif benar-benar memakainya.

        Sebelumnya pemuatan dijaga oleh ``reranker_enabled`` saja, tanpa memeriksa
        mode. Akibatnya model sebesar BAAI/bge-reranker-v2-m3 (XLM-RoBERTa-large)
        diunduh dan dimuat pada SETIAP startup, termasuk ketika jalur yang berjalan
        tidak pernah memanggilnya — cukup untuk mematikan instans 512 MB.

        ``_rerank`` dipanggil dari jalur leksikal dan hibrida. Jalur vektor murni
        mengembalikan hasil MMR apa adanya, jadi tidak memerlukan reranker.
        """
        if not self.reranker_enabled:
            return

        if self.retrieval_mode not in ("bm25", "hibrida"):
            logger.info(
                "Reranker dilewati: mode=%s tidak memanggil _rerank",
                self.retrieval_mode,
            )
            return

        try:
            if self.reranker_backend == "flashrank":
                from flashrank import Ranker

                # cache_dir WAJIB diisi: bawaan FlashRank adalah "/tmp", yang tidak
                # bermakna di Windows dan tidak bertahan antarrestart di kontainer.
                from pathlib import Path as _Path
                cache_dir = str(_Path(self.persist_dir).parent / "_flashrank")

                self._reranker = Ranker(
                    model_name=self.reranker_model,
                    cache_dir=cache_dir,
                    max_length=self.reranker_max_length,
                    log_level="WARNING",
                )
            else:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(
                    self.reranker_model,
                    max_length=self.reranker_max_length,
                )

            logger.info(
                "Reranker loaded — backend=%s model=%s candidate_k=%d mode=%s",
                self.reranker_backend,
                self.reranker_model,
                self.reranker_candidate_k,
                self.retrieval_mode,
            )

        except Exception as exc:
            self._reranker_error = f"{type(exc).__name__}: {exc}"
            self._reranker = None

            logger.warning(
                "Reranker gagal dimuat: %s",
                self._reranker_error,
            )

    def _bangun_bm25(self) -> None:
        """Bangun indeks BM25 atas korpus yang SAMA dengan indeks vektor.

        Dokumen diambil dari koleksi ChromaDB, BUKAN dibaca ulang dari PDF. Bila
        keduanya dibangun dari sumber berbeda, selisih hasil dapat berasal dari
        perbedaan korpus alih-alih dari cara menelusurinya — dan itu tidak akan
        terlihat dari angka mana pun.

        Memakai _tokenize() yang sama dengan SimpleKeywordRetriever: angka
        dipertahankan, karena ambang seperti "70" dan "180" justru sinyal yang
        diharapkan ditangkap pencocokan leksikal.

        KETERBATASAN yang diketahui: belum ada pemenggalan imbuhan bahasa
        Indonesia, sehingga "pemberian", "diberikan", dan "berikan" dihitung
        sebagai tiga token berbeda. Lee dkk. (2024) menunjukkan pilihan tokenizer
        berpengaruh pada pedoman diabetes berbahasa Korea; padanan Indonesianya
        belum diuji.
        """
        try:
            import chromadb
            from rank_bm25 import BM25Okapi

            col = chromadb.PersistentClient(path=self.persist_dir).get_collection(
                self.collection_name)
            # Dokumen DAN metadata diambil satu kali di sini, lalu dipakai ulang dari
            # memori oleh _hasil_dari_indeks. Sebelumnya metadata dibaca ulang dari
            # sqlite pada SETIAP kueri — seluruh korpus dimuat hanya untuk menyusun
            # lima baris hasil.
            r = col.get(include=["documents", "metadatas"])
            self._bm25_teks = r["documents"]
            self._bm25_metas = r["metadatas"]
            self._bm25_peta = {t: i for i, t in enumerate(self._bm25_teks)}
            self._bm25 = BM25Okapi([_tokenize(t) for t in self._bm25_teks])
            logger.info("Indeks BM25 dibangun atas %d potongan (mode=%s)",
                        len(self._bm25_teks), self.retrieval_mode)
        except Exception as exc:  # noqa: BLE001
            # Gagal membangun BM25 TIDAK boleh mematikan retriever: jalur vektor
            # tetap sah. Tetapi mode diturunkan secara EKSPLISIT dan dicatat, supaya
            # sistem tidak diam-diam menelusur dengan cara yang berbeda dari yang
            # dinyatakan config — persis jenis penyimpangan senyap yang menjadi
            # pokok penyelidikan T7-T14.
            logger.warning("Indeks BM25 gagal dibangun (%s); mode %s -> vektor",
                           exc, self.retrieval_mode)
            # Sebabnya DISIMPAN, bukan hanya dicatat ke log. Log mudah terlewat,
            # sedangkan penurunan mode mengubah cara sistem menelusur — operator
            # dan tes harus dapat menanyakannya langsung.
            self._bm25_error = f"{type(exc).__name__}: {exc}"
            self.mode_diminta = self.retrieval_mode
            self.retrieval_mode = "vektor"
            self._bm25 = None

    def _peringkat_bm25(self, query: str, n: int) -> List[int]:
        """Indeks korpus terurut menurun menurut skor BM25."""
        import numpy as np
        skor = self._bm25.get_scores(_tokenize(query))
        return list(np.argsort(-skor)[:n])

    def _gabung_rrf(self, daftar: List[List[int]]) -> List[int]:
        """Reciprocal Rank Fusion atas beberapa daftar indeks terurut.

        RRF dipilih karena bekerja pada PERINGKAT, bukan skor mentah: skor kosinus
        dan skor BM25 berada pada skala yang sama sekali berbeda dan tidak dapat
        dijumlahkan. Menormalkannya lebih dulu akan menambah satu parameter bebas
        yang harus disetel. Xiong dkk. (2024) memakai RRF untuk alasan yang sama
        pada benchmark MedRAG.
        """
        skor: Dict[int, float] = {}
        for satu in daftar:
            for pos, idx in enumerate(satu, start=1):
                skor[idx] = skor.get(idx, 0.0) + 1.0 / (self.rrf_k + pos)
        return [i for i, _ in sorted(skor.items(), key=lambda kv: -kv[1])]

    def _hasil_dari_indeks(self, indeks: List[int]) -> List[Dict[str, Any]]:
        """Susun baris hasil dari indeks korpus, lengkap dengan metadatanya.

        Korpus dibaca dari salinan memori yang disiapkan ``_bangun_bm25``. Indeks yang
        dipakai di sini BERASAL dari daftar yang sama, sehingga keduanya wajib merujuk
        urutan korpus yang identik — karena itu tidak boleh dibaca ulang dari sqlite
        secara terpisah.
        """
        docs, metas = self._bm25_teks, self._bm25_metas
        if not docs:
            # Jalur pemulihan: dipanggil tanpa indeks BM25 terbangun.
            import chromadb
            col = chromadb.PersistentClient(path=self.persist_dir).get_collection(
                self.collection_name)
            r = col.get(include=["documents", "metadatas"])
            docs, metas = r["documents"], r["metadatas"]
        keluar: List[Dict[str, Any]] = []
        for peringkat, i in enumerate(indeks, start=1):
            meta = dict(metas[i] or {})
            keluar.append({
                "rank": peringkat,
                "text": docs[i],
                "source": meta.get("source", "manual_kb"),
                # Skor kemiripan SENGAJA None pada jalur leksikal dan hibrida:
                # peringkat RRF bukan kemiripan kosinus, dan menampilkannya sebagai
                # "kemiripan" kepada dokter akan menyesatkan.
                "similarity": None,
                "metadata": meta,
            })
        return keluar

    def _skor_reranker(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[float]:
        """Skor relevansi kueri-terhadap-kandidat, searah dengan urutan ``candidates``.

        Dua backend menghitungnya dengan cara berbeda, tetapi keduanya WAJIB
        mengembalikan daftar sepanjang dan seurutan ``candidates`` — pemanggilnya
        memasangkan skor dengan kandidat lewat ``zip``. FlashRank mengembalikan
        kandidat yang sudah TERURUT, sehingga pemetaan balik ke urutan semula
        dilakukan di sini, bukan diserahkan ke pemanggil.
        """
        teks = [str(candidate.get("text", "")) for candidate in candidates]

        if self.reranker_backend == "flashrank":
            from flashrank import RerankRequest

            passages = [
                {"id": idx, "text": isi}
                for idx, isi in enumerate(teks)
            ]

            hasil = self._reranker.rerank(
                RerankRequest(query=query, passages=passages)
            )

            skor = [0.0] * len(teks)
            for baris in hasil:
                skor[int(baris["id"])] = float(baris.get("score", 0.0))
            return skor

        return [
            float(nilai)
            for nilai in self._reranker.predict(
                [[query, isi] for isi in teks],
                batch_size=8,
                show_progress_bar=False,
            )
        ]

    def _rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Susun ulang kandidat retrieval dengan cross-encoder.

        Alur:
            BM25 (leksikal)   -> candidate_k -> cross-encoder -> top_k
            BM25+Vektor (RRF) -> candidate_k -> cross-encoder -> top_k

        Reranker hanya mengurutkan kandidat yang sudah ditemukan tahap penelusuran.
        Ia TIDAK mencari dokumen baru, sehingga dokumen yang tidak masuk kolam
        kandidat tetap tak terjangkau berapa pun mutu rerankernya.

        Backend yang dipakai ditentukan ``reranker.backend``; lihat
        ``_skor_reranker``.
        """
        if not candidates:
            return []

        if self._reranker is None:
            logger.warning(
                "Reranker tidak tersedia; mengembalikan kandidat berdasarkan "
                "urutan retrieval."
            )
            return candidates[:top_k]

        try:
            scores = self._skor_reranker(query, candidates)

            # Tambahkan skor reranker ke setiap kandidat.
            reranked = []

            for candidate, score in zip(candidates, scores):
                row = dict(candidate)
                row["reranker_score"] = float(score)
                reranked.append(row)

            # Kedua backend sepakat: skor lebih tinggi = lebih relevan.
            reranked.sort(
                key=lambda item: item["reranker_score"],
                reverse=True,
            )

            # Rank diperbarui SETELAH reranking.
            for rank, row in enumerate(reranked[:top_k], start=1):
                row["rank"] = rank

            return reranked[:top_k]

        except Exception as exc:
            logger.warning(
                "Reranking gagal; menggunakan urutan retrieval: %s",
                exc,
            )

            # Jangan membuat seluruh RAG mati hanya karena reranker gagal.
            fallback = candidates[:top_k]

            for rank, row in enumerate(fallback, start=1):
                row["rank"] = rank

            return fallback

    @property
    def is_ready(self) -> bool:
        """Siap bila ADA jalur penelusuran yang dapat melayani kueri.

        Mode leksikal murni sengaja tidak membangun ruang vektor, sehingga syarat
        lama ``_vector_store is not None`` akan menyatakan retriever tidak siap dan
        membuat RAGPipeline mundur DIAM-DIAM ke potongan cadangan yang jauh lebih
        sempit daripada korpus penuh.
        """
        return self._vector_store is not None or self._bm25 is not None

    @staticmethod
    def _doc_key(doc: Any) -> str:
        """Kunci identitas dokumen untuk menggabungkan hasil MMR dengan skornya."""
        meta = dict(getattr(doc, "metadata", None) or {})
        chunk_id = meta.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        return f"__text__{hash(getattr(doc, 'page_content', ''))}"

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Ambil dokumen dengan MMR, sertai skor relevansi kueri-terhadap-chunk.

        MMR memilih k dokumen dari kolam kandidat fetch_k yang sama dengan yang
        dienumerasi kueri penilaian, sehingga setiap dokumen terpilih dijamin ada
        di dalam kolam skor. Vektor kueri dihitung SEKALI dan dipakai ulang oleh
        kedua pemanggilan, jadi tidak ada biaya embedding tambahan.

        CATATAN: skor yang dikembalikan adalah relevansi kueri-terhadap-chunk,
        BUKAN objektif MMR (yang mengurangi penalti redundansi). Karena itu urutan
        rank tidak selalu menurun monoton terhadap skor.
        """
        if not self.is_ready:
            raise RuntimeError(f"MMR retriever is not ready: {self._init_error}")

        top_k = top_k if top_k is not None else self.top_k
        fetch_k = self.fetch_k
        lambda_mult = self.lambda_mult

        # ── T13/T14: jalur leksikal dan hibrida ──────────────────────────────
        # Ditempatkan SEBELUM jalur vektor dan keluar lebih awal, supaya kode jalur
        # vektor di bawahnya tidak berubah satu baris pun dan angka era-vektor tetap
        # dapat direproduksi persis.
        if self._bm25 is not None and self.retrieval_mode in ("bm25", "hibrida"):
            # Kolam BM25 jalur hibrida SENGAJA tetap rrf_pool. Mengubahnya akan
            # mengubah masukan RRF dan membuat angka T13/T14 tidak dapat direproduksi.
            # Jalur leksikal mengambil kolamnya sendiri di bawah.
            urut_bm = self._peringkat_bm25(
                query,
                self.rrf_pool,
            )

            if self.retrieval_mode == "bm25":
                # LEXICAL RETRIEVAL
                # BM25 -> candidate pool -> cross-encoder -> top_k
                #
                # Sebelumnya cabang ini mengembalikan urut_bm[:top_k] apa adanya,
                # sehingga _rerank TIDAK PERNAH dipanggil pada jalur produksi meski
                # reranker.enabled bernilai true. Kolam kandidat kini disusun sama
                # seperti jalur hibrida agar cross-encoder dapat mengangkat dokumen
                # relevan yang terlempar jauh oleh pencocokan leksikal.
                if self._reranker is None:
                    # Tanpa reranker, perilaku sama persis dengan sebelumnya.
                    return self._hasil_dari_indeks(urut_bm[:top_k])

                candidate_k = max(top_k, self.reranker_candidate_k)

                # Kolam diambil ulang bila candidate_k melampaui rrf_pool, supaya
                # kandidat tidak terpotong diam-diam oleh ukuran kolam jalur hibrida.
                urut_kandidat = (
                    urut_bm
                    if candidate_k <= len(urut_bm)
                    else self._peringkat_bm25(query, candidate_k)
                )

                candidates = self._hasil_dari_indeks(
                    urut_kandidat[:candidate_k]
                )

                return self._rerank(
                    query,
                    candidates,
                    top_k,
                )

            # HYBRID RETRIEVAL
            # BM25 + Vector/MMR -> RRF

            docs_v = self._vector_store.max_marginal_relevance_search(
                query,
                k=self.rrf_pool,
                fetch_k=max(self.rrf_pool, fetch_k),
                lambda_mult=lambda_mult,
                filter=metadata_filter,
            )

            urut_v = [
                self._bm25_peta[d.page_content]
                for d in docs_v
                if d.page_content in self._bm25_peta
            ]

            # Gabungkan ranking BM25 dan Vector menggunakan RRF.
            rrf_indices = self._gabung_rrf(
                [urut_v, urut_bm]
            )

            # CANDIDATE POOL

            candidate_k = max(
                top_k,
                self.reranker_candidate_k,
            )

            candidate_indices = rrf_indices[:candidate_k]

            candidates = self._hasil_dari_indeks(
                candidate_indices
            )

            # RERANKING

            if self.reranker_enabled and self._reranker is not None:
                return self._rerank(
                    query=query,
                    candidates=candidates,
                    top_k=top_k,
                )

            # Jika reranker disabled/gagal load,
            # sistem tetap menggunakan hasil hybrid.
            return candidates[:top_k]

        kwargs: Dict[str, Any] = {"k": top_k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
        if metadata_filter:
            kwargs["filter"] = metadata_filter

        score_by_key: Dict[str, float] = {}
        docs = None

        if self._embeddings is not None:
            try:
                qvec = self._embeddings.embed_query(query)
                pool = self._vector_store.similarity_search_by_vector_with_relevance_scores(
                    embedding=qvec, k=fetch_k, filter=metadata_filter
                )
                score_by_key = {self._doc_key(d): float(s) for d, s in pool}
                docs = self._vector_store.max_marginal_relevance_search_by_vector(
                    embedding=qvec, k=top_k, fetch_k=fetch_k,
                    lambda_mult=lambda_mult, filter=metadata_filter,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Penilaian skor gagal, kembali ke MMR tanpa skor: %s", exc)
                docs = None

        if docs is None:
            retriever = self._vector_store.as_retriever(search_type="mmr", search_kwargs=kwargs)
            docs = retriever.invoke(query)

        results: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs, start=1):
            metadata = dict(doc.metadata or {})
            results.append(
                {
                    "rank": idx,
                    "text": doc.page_content,
                    "source": metadata.get("source", "manual_kb"),
                    "similarity": score_by_key.get(self._doc_key(doc)),
                    "metadata": metadata,
                }
            )

        return results

    def retrieve_with_context(
        self,
        query: str,
        patient_state: Dict[str, Any],
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        condition_glucose: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Ambil dokumen dengan kueri yang diperkaya tag kondisi.

        ``condition_glucose`` menentukan SUMBER kondisi secara eksplisit:
        - mode prediction-conditioned -> nilai glukosa TERPREDIKSI
        - mode standard               -> nilai glukosa SAAT INI

        Bila tidak diberikan, tag kondisi TIDAK ditambahkan sama sekali. Ini
        disengaja: menebak sumbernya (dulu selalu ``current_glucose``) membuat
        kueri diam-diam membawa kondisi yang sedang berlaku, padahal sistem
        mengklaim pengondisian pada kondisi terprediksi.
        """
        enhanced_query = self._enhance_query(
            query, patient_state, condition_glucose=condition_glucose
        )
        return self.retrieve(query=enhanced_query, top_k=top_k, metadata_filter=metadata_filter)

    def _enhance_query(
        self,
        query: str,
        patient_state: Dict[str, Any],
        condition_glucose: Optional[float] = None,
    ) -> str:
        """Bangun query retrieval dengan clinical intent sebagai anchor.

        Prediction/state hanya digunakan untuk menghasilkan condition tag
        yang ringkas. Nilai numerik dan narasi prediction tidak ditempel
        ke query karena dapat mendominasi retrieval.
        """
        from src.constants import CLASS_NORMAL, classify_glucose_3class

        conditions: List[str] = []

        if condition_glucose is not None:
            kondisi = classify_glucose_3class(float(condition_glucose))

            if kondisi != CLASS_NORMAL:
                conditions.append(kondisi)

        # Cabang "stres tinggi" DICABUT 24 Agustus 2026 bersama variabel stres.
        # Pencabutan ini TIDAK mengubah kueri produksi: sejak stres tak lagi ada
        # pada patient_state, `.get("stress_level", 0)` selalu 0 dan cabangnya
        # tidak pernah terpicu. Penting dicatat karena kueri retrieval sengaja
        # dibekukan menjelang evaluasi dokter agar hasil sebelum/sesudah sebanding.

        if not conditions:
            return query.strip()

        return f"{query.strip()} Konteks klinis: {', '.join(conditions)}."


def _metadata_matches(metadata: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
    for key, expected in metadata_filter.items():
        value = metadata.get(key)
        if isinstance(value, list):
            if expected not in value:
                return False
        else:
            if value != expected:
                return False
    return True


# Backward compatibility for existing imports.
DocumentRetriever = MMRRetriever
