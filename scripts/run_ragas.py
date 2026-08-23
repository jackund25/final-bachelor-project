#!/usr/bin/env python3
"""Evaluasi RAGAS atas basis data terkontrol (evaluation/ragas_dataset.json).

Berbeda dari scripts/ragas_eval.py yang mengevaluasi korpus produksi penuh, skrip
ini memakai koleksi TERPISAH berisi 2 halaman saja, sehingga:
  - ground truth dapat diverifikasi manual sampai ke kalimat sumbernya;
  - jumlah panggilan LLM penilai terkendali dan dapat diperkirakan di muka.

Pengaman kuota:
  --limit N        batasi jumlah sampel
  --metrics a,b    pilih metrik yang dijalankan
  --dry-run        cetak estimasi lalu berhenti, tanpa memanggil LLM
  cache            sampel yang sudah dinilai tidak dinilai ulang
  konfirmasi       estimasi panggilan dicetak dan meminta persetujuan sebelum jalan

Kuota per model dibaca ``kuota_model()``, bukan ditulis di sini, sebab nilainya berbeda
antar-model dan berubah sewaktu-waktu. Menurunkan laju TIDAK menambah kuota harian.

Jalankan:
    PYTHONPATH=. python scripts/run_ragas.py --dry-run
    PYTHONPATH=. python scripts/run_ragas.py --limit 2 --metrics context_recall
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch sebelum numpy — WinError 1114)

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "evaluation/ragas_dataset.json"
OUT_DIR = ROOT / "results/ragas"
CACHE_DIR = OUT_DIR / "cache"

EVAL_PERSIST = "models/chroma_db_eval"
EVAL_COLLECTION = "diabetes_kb_eval"

# Berapa panggilan LLM yang dipakai tiap metrik per sampel (ragas 0.2.6).
# answer_relevancy memakai strictness=3 -> 3 panggilan.
# context_precision menilai tiap konteks terambil -> sebanyak top_k.
# faithfulness: 1 ekstraksi pernyataan + 1 verdict NLI (perkiraan; bergantung
# panjang jawaban, jadi angkanya perkiraan bawah).
CALLS_PER_SAMPLE = {
    "faithfulness": 2,
    "answer_relevancy": 3,
    "context_precision": None,   # = top_k, diisi saat runtime
    "context_recall": 1,
}
DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Nilai glukosa wakil tiap kondisi, dipakai membangun kueri terkondisi. Ditetapkan
# sebelum eksekusi dan tidak disesuaikan setelah melihat hasil. Ketiganya diambil di
# TENGAH kelasnya, jauh dari ambang kelas maupun ambang "kritis", supaya klasifikasinya
# tidak sensitif terhadap pembulatan.
GLUKOSA_PER_KONDISI = {"hipoglikemia": 58.0, "normal": 120.0, "hiperglikemia": 230.0}

# Kuota berlaku PER MODEL, dan selisihnya antargenerasi mencapai 25 kali lipat; memakai
# satu angka untuk semua model membuat estimasi panggilan salah besar.
# Format: nama_model -> (RPM, RPD)
KUOTA_PER_MODEL = {
    "gemini-3.5-flash-lite": (15, 500),
    "gemini-3.1-flash-lite": (15, 500),
    "gemini-2.5-flash-lite": (10, 20),
    "gemini-2.5-flash": (5, 20),
    "gemini-3-flash": (5, 20),
}
# Model yang tidak terdaftar diperlakukan konservatif: pakai kuota terkecil yang diketahui,
# supaya estimasi tidak pernah terlalu optimistis pada model yang belum diukur.
KUOTA_TIDAK_DIKENAL = (5, 20)


def kuota_model(nama: str) -> tuple[int, int]:
    """(RPM, RPD) untuk satu model. Konservatif bila model belum terdaftar."""
    return KUOTA_PER_MODEL.get(nama, KUOTA_TIDAK_DIKENAL)

# Frasa disclaimer yang dipaksa RAGPipeline._ensure_disclaimer().
DISCLAIMER_FRASA = "keputusan medis final tetap pada dokter"


def _patient_state(pred: float) -> Dict[str, Any]:
    """Keadaan pasien minimal namun lengkap untuk PredictionConditionedQueryBuilder.

    Nilai non-glukosa sengaja netral (stres 5, aktivitas 20, IOB 0, COB 0) supaya kueri
    tidak membawa faktor kontribusi yang tidak ada di dataset dan tidak dapat diverifikasi.
    """
    return {"current_glucose": pred, "stress_level": 5, "activity_level": 20,
            "insulin_on_board": 0.0, "carbs_on_board": 0.0}


# Kalimat yang ditambahkan RAGPipeline._ensure_disclaimer() bila LLM tidak menulisnya.
_SUFIKS_SISTEM = "Catatan: Keputusan medis final tetap pada dokter."

# Penanda disclaimer yang ditulis LLM sendiri, bukan yang ditambahkan sistem. Ditulis
# longgar dengan sengaja: pola yang menyasar heading persis lolos ketika model menulis
# variasi seperti "Disclaimer Dokter:". Karena teks disclaimer selalu berada di AKHIR
# jawaban, pola pertama menyapu dari kemunculan kata itu sampai habis.
_POLA_DISCLAIMER_LLM = [
    # Hanya 'Disclaimer' sebagai JUDUL BLOK di awal baris. Sengaja TIDAK menyapu kata itu
    # di tengah kalimat: pola yang terlalu longgar memotong kalimat sah seperti "Jawab
    # tanpa disclaimer sama sekali" menjadi "Jawab tanpa", yakni merusak data tanpa tanda.
    r"(?ms)^[ \t]*(?:\*{1,2}|#{1,4}|-)?\s*Disclaimer\b.*",
    r"(?mi)^\s*(Catatan|Penting|Perhatian)\s*:\s*[^\n]*tidak menggantikan[^\n]*$",
    r"[^.!?\n]*tidak menggantikan penilaian klinis[^.!?]*[.!?]",
    r"[^.!?\n]*keputusan medis final tetap pada dokter[^.!?]*[.!?]",
    r"[^.!?\n]*bersifat (umum|panduan)[^.!?]*tidak menggantikan[^.!?]*[.!?]",
]


def _tulis_disclaimer_sendiri(teks_utuh: str) -> bool:
    """Apakah LLM menulis disclaimer SENDIRI, bukan hasil penegakan sistem?

    Diperiksa dengan membuang sufiks sistem lebih dulu. Tanpa langkah ini, pemeriksaan
    selalu bernilai True menurut konstruksi, karena _ensure_disclaimer() menambahkan
    frasa wajib bila belum ada — dan angka "100% kepatuhan" menjadi tidak bermakna.
    """
    import re as _re
    tanpa_sufiks = teks_utuh.strip()
    if tanpa_sufiks.endswith(_SUFIKS_SISTEM):
        tanpa_sufiks = tanpa_sufiks[: -len(_SUFIKS_SISTEM)].strip()
    return bool(_re.search(r"disclaimer|tidak menggantikan penilaian klinis",
                           tanpa_sufiks, _re.IGNORECASE))


def _buang_disclaimer(teks: str) -> str:
    """Hapus SELURUH disclaimer — sufiks sistem maupun paragraf buatan LLM."""
    import re as _re
    bersih = teks.strip()
    for pola in _POLA_DISCLAIMER_LLM:
        bersih = _re.sub(pola, "", bersih, flags=_re.IGNORECASE | _re.DOTALL)
    return _re.sub(r"\n{3,}", "\n\n", bersih).strip()


def _load_cases(dataset_path: Path, limit: int | None) -> List[Dict[str, Any]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = [c for c in data["kasus"] if (c.get("pertanyaan") or "").strip()]
    if not cases:
        raise SystemExit(
            f"Tidak ada kasus terisi di {dataset_path}.\n"
            f"Lengkapi medan 'pertanyaan', 'jawaban_acuan', dan 'konteks_acuan' terlebih dulu."
        )
    return cases[:limit] if limit else cases


def _hitung_pernyataan(result: Any, ids: List[str]) -> Dict[str, Any]:
    """Jumlah pernyataan yang diekstraksi RAGAS per sampel saat menilai faithfulness.

    RAGAS 0.2.6 menyimpan jejak antara pada `ragas_traces`. Strukturnya tidak dijamin
    stabil antarversi, jadi penelusurannya defensif: bila apa pun meleset, nilainya
    `None` dan ketidaktersediaannya DICATAT alih-alih ditebak.
    """
    hasil: Dict[str, Any] = {i: None for i in ids}
    traces = getattr(result, "ragas_traces", None) or getattr(result, "traces", None)
    if not traces:
        return hasil
    try:
        runs = list(traces.values()) if isinstance(traces, dict) else list(traces)
        for idx, run in enumerate(runs):
            if idx >= len(ids):
                break
            n = None
            for atribut in ("children", "outputs", "output"):
                simpul = getattr(run, atribut, None) or (
                    run.get(atribut) if isinstance(run, dict) else None)
                if simpul is None:
                    continue
                teks = json.dumps(simpul, default=str)
                # NLIStatementOutput memuat daftar `statements`; hitung elemennya.
                cocok = re.findall(r'"statement"\s*:', teks)
                if cocok:
                    n = len(cocok)
                    break
            hasil[ids[idx]] = n
    except Exception:  # noqa: BLE001 — ketidaktersediaan bukan kegagalan evaluasi
        return {i: None for i in ids}
    return hasil


def _cache_key(case_id: str, metric: str, judge_model: str, top_k: int) -> str:
    raw = f"{case_id}|{metric}|{judge_model}|k={top_k}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _estimate(cases: List[Dict[str, Any]], metrics: List[str], top_k: int,
              cached: int) -> Dict[str, Any]:
    per_metric: Dict[str, int] = {}
    for m in metrics:
        per = CALLS_PER_SAMPLE.get(m)
        per_metric[m] = (top_k if per is None else per) * len(cases)
    generasi = len(cases)  # 1 panggilan pembangkitan jawaban per sampel
    total = generasi + sum(per_metric.values()) - cached
    return {
        "n_sampel": len(cases),
        "metrik": metrics,
        "top_k": top_k,
        "panggilan_pembangkitan": generasi,
        "panggilan_per_metrik": per_metric,
        "sudah_ter-cache": cached,
        "total_estimasi": max(total, 0),
    }


def _project(n: int, metrics: List[str], top_k: int) -> int:
    """Proyeksi jumlah panggilan bila n kasus terisi penuh."""
    total = n  # pembangkitan jawaban
    for m in metrics:
        per = CALLS_PER_SAMPLE.get(m)
        total += (top_k if per is None else per) * n
    return total


def _print_estimate(est: Dict[str, Any], rpd: int = 20, rpm: int = 10,
                    model: str = "?") -> None:
    print("=" * 66)
    print("ESTIMASI PANGGILAN LLM SEBELUM EKSEKUSI")
    print("=" * 66)
    print(f"  Sampel dievaluasi     : {est['n_sampel']}")
    print(f"  Metrik                : {', '.join(est['metrik'])}")
    print(f"  top_k konteks         : {est['top_k']}")
    print(f"  Pembangkitan jawaban  : {est['panggilan_pembangkitan']}")
    for m, n in est["panggilan_per_metrik"].items():
        print(f"  Metrik {m:<20s}: {n}")
    if est["sudah_ter-cache"]:
        print(f"  Sudah ter-cache (dilewati): -{est['sudah_ter-cache']}")
    print("-" * 66)
    total = est["total_estimasi"]
    print(f"  TOTAL ESTIMASI        : ~{total} panggilan")
    print("=" * 66)
    penuh = est.get("proyeksi_dataset_penuh")
    if penuh and penuh["n"] != est["n_sampel"]:
        print(f"  Proyeksi bila {penuh['n']} kasus terisi penuh: ~{penuh['total']} panggilan")
        print("=" * 66)

    rps = float(os.getenv("RAGAS_RPS", "0.13"))
    menit_min = total / max(rpm, 1)
    menit_setelan = total / (rps * 60) if rps > 0 else float("inf")
    hari = -(-total // max(rpd, 1))  # pembulatan ke atas
    print(f"  KUOTA PER MODEL — berlaku PER MODEL, bukan per akun:")
    print(f"    model                : {model}")
    print(f"    batas laju           : {rpm} permintaan/menit")
    print(f"    KUOTA HARIAN         : {rpd} permintaan/hari")
    print(f"    setelan RAGAS_RPS    : {rps} req/dtk (~{rps * 60:.1f}/menit)")
    print(f"    waktu minimum        : ~{menit_min:.0f} mnt (pada batas laju)")
    print(f"    waktu pada setelan   : ~{menit_setelan:.0f} mnt")
    print(f"    perlu                : ~{hari} HARI untuk {total} panggilan")
    print("=" * 66)
    if hari > 1:
        lebih_besar = [m for m, (_, d) in KUOTA_PER_MODEL.items() if d > rpd]
        print(f"  PERINGATAN: {total} panggilan MELEBIHI kuota harian {rpd}.")
        print(f"  Menurunkan laju TIDAK menambah kuota harian.")
        if lebih_besar:
            print(f"  Model dengan kuota lebih besar: {', '.join(sorted(lebih_besar))}")
        print("=" * 66)
    print("Catatan: jumlah faithfulness bergantung panjang jawaban, jadi angka di")
    print("atas adalah perkiraan BAWAH.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluasi RAGAS atas basis data terkontrol")
    ap.add_argument("--limit", type=int, default=None, help="Batasi jumlah sampel")
    ap.add_argument("--dataset", default=str(DATASET),
                    help="Path dataset evaluasi JSON")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                    help=f"Metrik dipisah koma. Pilihan: {', '.join(DEFAULT_METRICS)}")
    ap.add_argument("--dry-run", action="store_true",
                    help="Cetak estimasi lalu berhenti, tanpa memanggil LLM")
    ap.add_argument("--yes", action="store_true", help="Lewati prompt konfirmasi")
    ap.add_argument("--no-cache", action="store_true", help="Abaikan cache penilaian")
    ap.add_argument("--top-k", type=int, default=None, help="Override top_k retrieval")
    ap.add_argument("--persist", default=EVAL_PERSIST)
    ap.add_argument("--collection", default=EVAL_COLLECTION)
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    tidak_dikenal = [m for m in metrics if m not in DEFAULT_METRICS]
    if tidak_dikenal:
        print(f"GAGAL: metrik tidak dikenal: {', '.join(tidak_dikenal)}\n"
              f"Pilihan: {', '.join(DEFAULT_METRICS)}", file=sys.stderr)
        return 1

    from src.config import load_rag_config
    cfg = load_rag_config()
    top_k = args.top_k or cfg.top_k
    judge_model = os.getenv("RAGAS_JUDGE_MODEL") or cfg.llm_model

    cases = _load_cases(Path(args.dataset), args.limit)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = 0
    if not args.no_cache:
        for c in cases:
            for m in metrics:
                if _cache_path(_cache_key(c["id"], m, judge_model, top_k)).exists():
                    cached += 1

    # Proyeksi untuk seluruh kerangka kasus (termasuk yang belum diisi), supaya
    # anggaran kuota dapat direncanakan sebelum dataset dilengkapi.
    n_kerangka = len(json.loads(DATASET.read_text(encoding="utf-8"))["kasus"])
    est = _estimate(cases, metrics, top_k, cached)
    est["proyeksi_dataset_penuh"] = {
        "n": n_kerangka,
        "total": _project(n_kerangka, metrics, top_k),
    }
    _rpm, _rpd = kuota_model(judge_model)
    est["model"] = judge_model
    est["kuota_model"] = {"rpm": _rpm, "rpd": _rpd,
                          "terdaftar": judge_model in KUOTA_PER_MODEL}
    _print_estimate(est, rpd=_rpd, rpm=_rpm, model=judge_model)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "estimasi_terakhir.json").write_text(
        json.dumps(est, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print("\n--dry-run: berhenti tanpa memanggil LLM.")
        return 0

    if not args.yes:
        try:
            jawab = input("\nLanjutkan dan panggil LLM? [y/N] ").strip().lower()
        except EOFError:
            # Dijalankan non-interaktif (pipa/CI) tanpa --yes: perlakukan sebagai
            # "tidak". Default aman — jangan pernah membakar kuota tanpa persetujuan.
            print("\nTidak ada masukan (non-interaktif). Dibatalkan; "
                  "pakai --yes bila memang ingin menjalankannya.")
            return 0
        if jawab != "y":
            print("Dibatalkan; tidak ada panggilan LLM yang dilakukan.")
            return 0

    # ── Eksekusi ─────────────────────────────────────────────────
    if not Path(args.persist).exists():
        print(f"GAGAL: koleksi evaluasi belum dibangun di {args.persist}.\n"
              f"Jalankan: PYTHONPATH=. python scripts/build_eval_kb.py", file=sys.stderr)
        return 1

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    from src.rag.retriever import MMRRetriever

    retriever = MMRRetriever(persist_dir=args.persist, collection_name=args.collection)
    if not retriever.is_ready:
        print("GAGAL: retriever koleksi evaluasi tidak siap.", file=sys.stderr)
        return 1

    import nest_asyncio
    nest_asyncio.apply()

    from langchain_openai import ChatOpenAI
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    import ragas.metrics as M

    api_key = os.getenv("RAGAS_JUDGE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    base_url = os.getenv("RAGAS_JUDGE_BASE_URL",
                         "https://generativelanguage.googleapis.com/v1beta/openai/")
    if not api_key:
        print("GAGAL: GOOGLE_API_KEY tidak ditemukan di environment / .env", file=sys.stderr)
        return 1

    # Laju DITURUNKAN DARI KUOTA MODEL, bukan dipatok satu angka. Sebelumnya nilainya
    # hardcoded dan harus disunting tiap kali model berganti — sumber kesalahan yang sama
    # dengan RPD tunggal. Margin 80% dari batas laju supaya tidak menyentuh 429; retry
    # backoff justru memperlambat seluruh evaluasi (A4: satu permintaan menunggu 33 detik).
    _rpm_model, _ = kuota_model(judge_model)
    rate = float(os.getenv("RAGAS_RPS", str(round(_rpm_model * 0.8 / 60, 3))))
    print(f"Batas laju model {judge_model}: {_rpm_model} RPM -> "
          f"RAGAS_RPS={rate} ({rate * 60:.1f}/menit, margin 80%)")
    limiter = InMemoryRateLimiter(requests_per_second=rate, check_every_n_seconds=0.5,
                                  max_bucket_size=1)

    gen_llm = ChatOpenAI(model=os.getenv("RAGAS_GEN_MODEL", judge_model), api_key=api_key,
                         base_url=base_url, temperature=cfg.temperature,
                         max_tokens=cfg.max_tokens, max_retries=5, rate_limiter=limiter)
    judge = LangchainLLMWrapper(
        ChatOpenAI(model=judge_model, api_key=api_key, base_url=base_url, temperature=0.0,
                   max_tokens=4096, max_retries=5, rate_limiter=limiter))
    embed = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=cfg.embedding_model, model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}))

    METRIC_OBJ = {
        "faithfulness": M.faithfulness,
        "answer_relevancy": M.answer_relevancy,
        "context_precision": M.context_precision,
        "context_recall": M.context_recall,
    }

    # ── Pembangkitan lewat JALUR PRODUKSI ──────────────────────────────────────
    # Versi sebelumnya membangun prompt minimal sendiri dan memanggil gen_llm
    # langsung, sehingga RAGPipeline TIDAK PERNAH dipakai. Akibatnya evaluasi
    # mengukur LLM telanjang, bukan sistem yang dibahas laporan: tanpa SYSTEM_PROMPT,
    # tanpa kueri terkondisi-prediksi, tanpa penanda sumber, dan medan
    # `kondisi_terprediksi` pada dataset tidak pernah terpakai sama sekali.
    from src.rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(chroma_persist_dir=args.persist, collection_name=args.collection,
                           llm_provider="gemini")
    pipeline.build()
    chain = getattr(pipeline.generator, "chain", None)
    if not (chain is not None and getattr(chain, "is_ready", False)):
        print("GAGAL: rantai LLM produksi tidak siap. Sistem akan menjawab dengan "
              "template tanpa error, dan skor RAGAS-nya tidak sah.", file=sys.stderr)
        return 1

    samples = []
    disclaimer_ada = patuh_sistem_n = sisa_disclaimer = 0
    for c in cases:
        pred = GLUKOSA_PER_KONDISI[c["kondisi_terprediksi"]]
        res = pipeline.answer(patient_state=_patient_state(pred), prediction=float(pred),
                              query=c["pertanyaan"], top_k=top_k)
        contexts = [d["text"] for d in res["retrieved_docs"]]
        jawaban_utuh = res["explanation"]

        # Kepatuhan disclaimer diperiksa dengan PENCOCOKAN TEKS BIASA — tanpa panggilan
        # Disclaimer dibuang sebelum penilaian: ia artefak sistem yang tetap, bukan
        # pernyataan tentang pasien, sehingga membiarkannya berarti mengukur disclaimer
        # alih-alih mutu pembangkitan. Kepatuhannya dilaporkan terpisah, pada dua
        # tingkat: model (LLM menulisnya sendiri) dan sistem (frasa wajib ada setelah
        # penegakan). Tingkat sistem selalu penuh menurut konstruksi; yang informatif
        # tingkat model.
        patuh_model = _tulis_disclaimer_sendiri(jawaban_utuh)
        patuh_sistem = DISCLAIMER_FRASA in jawaban_utuh.lower()
        disclaimer_ada += int(patuh_model)
        patuh_sistem_n += int(patuh_sistem)
        jawaban_dinilai = _buang_disclaimer(jawaban_utuh)
        sisa_disclaimer += int("disclaimer" in jawaban_dinilai.lower()
                               or "tidak menggantikan penilaian" in jawaban_dinilai.lower())

        samples.append(SingleTurnSample(
            user_input=c["pertanyaan"],
            response=jawaban_dinilai,
            retrieved_contexts=contexts,
            reference=c["jawaban_acuan"],
        ))

    n = max(len(cases), 1)
    print(f"\n=== Kepatuhan disclaimer (pencocokan teks, TANPA panggilan LLM) ===")
    print(f"  Tingkat MODEL  : {disclaimer_ada}/{len(cases)} = "
          f"{100.0 * disclaimer_ada / n:.1f}% menulis disclaimer sendiri")
    print(f"  Tingkat SISTEM : {patuh_sistem_n}/{len(cases)} = "
          f"{100.0 * patuh_sistem_n / n:.1f}% memuat frasa wajib setelah penegakan"
          f"  [bukti KNF-06]")
    print(f"  Catatan: tingkat SISTEM selalu 100% menurut konstruksi karena "
          f"_ensure_disclaimer() menambahkan frasa bila belum ada. Yang informatif "
          f"adalah tingkat MODEL.")
    # Penjagaan: bila disclaimer masih tersisa di teks yang dinilai, perlakuan yang
    # dinyatakan TIDAK terlaksana dan skor faithfulness ikut menilai disclaimer.
    if sisa_disclaimer:
        print(f"\n  PERINGATAN: {sisa_disclaimer}/{len(cases)} jawaban MASIH memuat sisa "
              f"disclaimer setelah pembersihan. Skor faithfulness ikut menilai teks "
              f"disclaimer — perlakuan yang dinyatakan TIDAK terlaksana sepenuhnya.")
    else:
        print(f"  Verifikasi: 0/{len(cases)} jawaban menyisakan disclaimer setelah "
              f"pembersihan — perlakuan terlaksana.")
    print("Disclaimer DIBUANG sebelum penilaian RAGAS — perlakuan disengaja dan dicatat, "
          "periksa urutan impor torch.")

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[METRIC_OBJ[m] for m in metrics],
        llm=judge, embeddings=embed,
        run_config=RunConfig(max_workers=1, timeout=180, max_retries=5, max_wait=60),
    )

    df = result.to_pandas()
    df.insert(0, "id", [c["id"] for c in cases])

    # JUMLAH PERNYATAAN yang diekstraksi RAGAS saat menghitung faithfulness.
    # Tanpa angka ini skor faithfulness TIDAK DAPAT DITAFSIRKAN: metrik itu adalah
    # rasio pernyataan-didukung terhadap total pernyataan, sehingga jawaban panjang
    # punya lebih banyak peluang gagal. Jawaban terpendek paling aman skornya.
    # Bila pustaka tidak memaparkannya, hal itu DICATAT, bukan didiamkan.
    n_pernyataan = _hitung_pernyataan(result, [c["id"] for c in cases])
    df["n_pernyataan_faithfulness"] = [n_pernyataan.get(i) for i in df["id"]]
    df["panjang_jawaban_char"] = [len(str(r)) for r in df.get("response", [""] * len(df))]

    df.to_csv(OUT_DIR / "per_sample.csv", index=False, encoding="utf-8")
    tersedia = sum(1 for v in n_pernyataan.values() if v is not None)
    print(f"\nJumlah pernyataan faithfulness: {tersedia}/{len(cases)} sampel tersedia"
          + ("" if tersedia else "  -> TIDAK TERSEDIA dari keluaran pustaka; dicatat apa adanya"))

    # Cache per (sampel, metrik) agar analisis ulang tidak memanggil LLM lagi.
    for _, row in df.iterrows():
        for m in metrics:
            if m not in row:
                continue
            key = _cache_key(row["id"], m, judge_model, top_k)
            _cache_path(key).write_text(
                json.dumps({"id": row["id"], "metric": m, "score": float(row[m])},
                           ensure_ascii=False), encoding="utf-8")

    ringkas = {m: float(df[m].mean()) for m in metrics if m in df.columns}
    (OUT_DIR / "summary.json").write_text(
        json.dumps({"n": len(cases), "metrik": ringkas, "judge_model": judge_model,
                    "top_k": top_k}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== HASIL ===")
    for m, v in ringkas.items():
        print(f"  {m:<22s}: {v:.3f}")
    print(f"\nKeluaran mentah -> {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
