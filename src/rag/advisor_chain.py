"""LCEL-based advisor chain for diabetes RAG — supports Gemini and Ollama backends."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .prompts import SYSTEM_PROMPT, build_question_payload, format_context_with_citations

logger = logging.getLogger(__name__)


class DiabetesAdvisorChain:
    """Generate clinical advisory text from retrieved context and patient state.

    Supports two LLM backends:
    - ``"gemini"``  — Google Gemini via langchain-google-genai (recommended, free tier)
    - ``"ollama"``  — local Ollama server (fallback for offline use)

    Falls back to a rule-based template if neither backend initialises successfully.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        config: Optional[Any] = None,
    ):
        from src.config import load_rag_config

        cfg = config or load_rag_config()
        provider = provider or cfg.llm_provider
        self.model_name = model_name or (
            cfg.llm_model if provider == "gemini" else cfg.ollama_llm_model
        )
        self.provider = provider
        self.ollama_base_url = ollama_base_url or cfg.ollama_base_url
        self.temperature = temperature if temperature is not None else cfg.temperature
        self.max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
        model_name = self.model_name
        self._chain = None
        self._init_error: Optional[str] = None

        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_PROMPT),
                    (
                        "human",
                        "Konteks dokumen:\n{context}\n\n"
                        "Data dan pertanyaan:\n{question_payload}\n\n"
                        "Berikan jawaban klinis ringkas yang langsung menjawab pertanyaan. "
                        "Berikan langkah aksi hanya jika pertanyaan meminta tindakan atau penanganan, "
                        "atau jika tindakan tersebut diperlukan untuk keselamatan pasien. "
                        "Jangan menambahkan rekomendasi klinis yang tidak diminta. "
                        "Selalu sertakan disclaimer dokter.",
                    ),
                ]
            )

            if provider == "gemini":
                api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GOOGLE_API_KEY environment variable not set. "
                        "Dapatkan API key gratis di https://aistudio.google.com/app/apikey"
                    )
                from langchain_google_genai import ChatGoogleGenerativeAI

                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            else:
                from langchain_ollama import ChatOllama

                llm = ChatOllama(
                    model=model_name,
                    base_url=self.ollama_base_url,
                    temperature=self.temperature,
                    num_predict=self.max_tokens,
                )

            self._chain = prompt | llm | StrOutputParser()
            logger.info("Advisor chain initialised — provider=%s model=%s", provider, model_name)

        except Exception as exc:
            self._init_error = str(exc)
            logger.warning("Advisor chain fallback active (provider=%s): %s", provider, exc)

    @property
    def is_ready(self) -> bool:
        return self._chain is not None

    def generate(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        patient_state: Dict[str, Any],
        prediction: float,
        horizon_minutes: Optional[int] = None,
        clinical_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        context_block = format_context_with_citations(retrieved_docs)
        question_payload = build_question_payload(
            query, patient_state, prediction, horizon_minutes=horizon_minutes
        )
        # Blok kondisi klinis terstruktur (tren, urgensi, IOB/COB) dari
        # PredictionConditionedQueryBuilder._llm_context(). Sebelumnya blok ini dibangun
        # lalu dibuang — tidak pernah sampai ke LLM sama sekali.
        if clinical_context:
            question_payload = f"{clinical_context}\n\n{question_payload}"

        if self._chain is not None:
            try:
                answer = self._chain.invoke(
                    {
                        "context": context_block,
                        "question_payload": question_payload,
                    }
                )
                return {
                    "answer": answer.strip(),
                    "sources": self._extract_sources(retrieved_docs),
                    "narasi_llm": True,
                }
            except Exception as exc:
                logger.warning("Advisor chain invocation failed, using template: %s", exc)

        # Jalur cadangan KEDUA. generator.py menangani kasus "rantai LLM tidak pernah
        # dibangun" (self._chain is None sejak __init__). Cabang ini menangani kasus
        # yang berbeda: rantai ADA tapi invoke() gagal saat runtime (kuota habis,
        # jaringan, respons tidak valid). Keduanya harus menghasilkan narasi_llm=False
        # dengan alasan yang sama — dokter tidak boleh mengira templat ini penalaran
        # LLM. Sebelumnya kunci ini tidak pernah disetel di sini, sehingga default
        # `True` pada pipeline.py membuat kegagalan runtime tampil seolah sukses.
        return {
            "answer": self._template_answer(patient_state, prediction, retrieved_docs),
            "sources": self._extract_sources(retrieved_docs),
            "narasi_llm": False,
        }

    @staticmethod
    def _extract_sources(retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ringkasan sumber terstruktur, LENGKAP dengan teks yang dikutip.

        Dijadikan ``staticmethod`` karena tidak bergantung pada keadaan rantai mana pun.
        Itu memungkinkan jalur cadangan di ``generator.py`` — yang berjalan justru ketika
        rantai LLM TIDAK ada — tetap menyusun daftar rujukan yang sama persis.

        Membaca skema metadata halaman yang baru (kb_id/lembaga/halaman_cetak) dengan
        fallback ke kunci lama (sumber/judul/halaman) agar chunk manual_kb tetap jalan.

        DIPERBARUI 24 Agustus 2026. Sebelumnya fungsi ini hanya mengembalikan lima medan
        identitas, sehingga antarmuka hanya dapat menampilkan "PERKENI · 2021 · hal. 33"
        tanpa satu pun kalimat yang dikutip. Dokter tidak dapat menilai apakah rekomendasi
        benar-benar berpijak pada dokumen — padahal keterlacakan bukti justru salah satu
        aspek yang mereka nilai.

        Pemotongan TIDAK dihitung di sini melainkan didelegasikan ke ``build_source_list``,
        yang memotong pada batas KALIMAT dengan empat penjaga (mis. agar "PERKENI 2021
        hal. 12" tidak terpotong menjadi "... hal.") dan diuji di
        ``tests/test_citations_snippet.py``. Menyalin aturannya ke tempat lain — apalagi
        ke TypeScript di frontend — akan memunculkan kembali cacat yang sudah ditutup,
        dan tesnya tidak akan menangkapnya.
        """
        from .citations import build_source_list

        output: List[Dict[str, Any]] = []

        for row in build_source_list(retrieved_docs):
            output.append(
                {
                    # --- Medan lama, dipertahankan agar pemanggil tidak putus ---
                    "source": row["lembaga"] or row["nama_dokumen"],
                    "title": row["judul_lengkap"] or row["nama_dokumen"],
                    "year": row["tahun"] or "N/A",
                    "page": row["page_label"],
                    "kb_id": row["kb_id"],
                    # --- Medan baru: isi kutipan dan provenansinya ---
                    "rank": row["rank"],
                    "nama_dokumen": row["nama_dokumen"],
                    # Kutipan tampilan, sudah berhenti di batas kalimat.
                    "snippet": row["snippet"],
                    # Potongan utuh, untuk sakelar "tampilkan secara utuh".
                    "teks_lengkap": row["teks_lengkap"],
                    "n_char": row["n_char"],
                    # Membedakan "potongan memang sependek itu" dari "tampilannya
                    # yang memotong" — pembedaan yang diperlukan saat memeriksa
                    # keluhan teks terpotong.
                    "snippet_terpotong": row["snippet_terpotong"],
                    "cara_potong": row["cara_potong"],
                    # Menggambarkan POTONGAN ASLI hasil pengindeksan, bukan tampilan.
                    "mulai_kalimat_utuh": row["mulai_kalimat_utuh"],
                    "akhir_kalimat_utuh": row["akhir_kalimat_utuh"],
                    "chunk_id": row["chunk_id"],
                }
            )

        return output

    def _template_answer(
        self,
        patient_state: Dict[str, Any],
        prediction: float,
        retrieved_docs: List[Dict[str, Any]],
    ) -> str:
        from src.constants import (
            CLASS_HYPER, CLASS_HYPO, classify_glucose_3class, classify_glucose_5zone,
            risk_label_id,
        )

        glucose = float(patient_state.get("current_glucose", 100.0))
        risk = risk_label_id(classify_glucose_5zone(prediction))
        kondisi = classify_glucose_3class(prediction)
        if kondisi == CLASS_HYPO:
            action = "Lakukan aturan 15-15 dan evaluasi medis segera."
        elif kondisi == CLASS_HYPER:
            action = "Perkuat hidrasi, evaluasi asupan, dan pantau ulang glukosa dalam 1 jam."
        else:
            action = "Pertahankan pola makan dan monitoring rutin."

        # SEBELUMNYA teks "1 jam" ditulis tetap di sini, tidak peduli horizon mana
        # yang sebenarnya diprediksi. `predicted_glucose` yang dikirim clinical.py
        # SENGAJA mengambil horizon TERDEKAT (mis. 30 menit, bukan 60), supaya risiko
        # jangka pendek — yang paling mendesak, terutama untuk hipoglikemia — menjadi
        # dasar advisory. Label tetap "1 jam" berarti dokter membaca "56.0 -> 67.9
        # mg/dL dalam 1 jam" padahal panel FORECAST di sebelahnya sudah menunjukkan
        # 67.9 adalah nilai 30 MENIT, bukan 60. Horizonnya sendiri sudah tersimpan di
        # patient_state["prediction_horizon_minutes"] sejak clinical.py — di sini
        # hanya perlu dibaca, bukan ditebak ulang.
        horizon = patient_state.get("prediction_horizon_minutes")
        horizon_label = (
            "1 jam" if horizon == 60
            else f"{int(horizon)} menit" if horizon is not None
            else "berikut"  # horizon tidak diketahui — jangan mengarang angka
        )

        source_count = len(retrieved_docs)
        # Kata-katanya HARUS sama persis dengan substring yang dicek
        # RAGPipeline._ensure_disclaimer() ("keputusan medis final tetap pada
        # dokter"). Sebelumnya kalimat di sini berbunyi "keputusan KLINIS final
        # tetap MEMERLUKAN PENILAIAN dokter" — beda kata, sehingga pengecekan
        # gagal mengenalinya sebagai disclaimer yang sudah ada, dan
        # _ensure_disclaimer menambahkan kalimat KEDUA di akhir jawaban.
        return (
            f"Status risiko: {risk}. Prediksi glukosa {horizon_label} adalah "
            f"{prediction:.1f} mg/dL dari kondisi saat ini {glucose:.1f} mg/dL. "
            f"Rekomendasi awal: {action} Konteks yang digunakan: {source_count} sumber. "
            "Catatan: keputusan medis final tetap pada dokter."
        )
