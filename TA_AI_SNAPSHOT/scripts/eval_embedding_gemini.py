"""T10 — kandidat produksi: embedding Gemini terkelola lawan MiniLM lokal.

PRAPENDAFTARAN, ditulis SEBELUM hasil dilihat.

APA YANG PERCOBAAN INI BISA DAN TIDAK BISA JAWAB — dibaca lebih dulu.

T8 dan T9 dirancang mengubah SATU faktor sehingga penyebabnya dapat dikaitkan.
Percobaan ini TIDAK demikian, dan itu disengaja. Berpindah ke gemini-embedding-2
mengubah sekurang-kurangnya EMPAT hal sekaligus:

  1. cakupan bahasa   : Inggris  -> multibahasa
  2. jendela token    : 256      -> 8192 (pemotongan hilang)
  3. dimensi vektor   : 384      -> 3072
  4. keluarga model   : MiniLM disuling -> model terkelola Google

Karena itu hasilnya TIDAK BOLEH dibaca sebagai "penyebabnya sudah ketemu".
Percobaan ini menjawab pertanyaan praktis yang berbeda: bila sistem berpindah ke
embedding terkelola, apakah retrievalnya membaik dan sepadan dengan biayanya?
Pemisahan penyebab diserahkan pada T8 (jendela) dan T9 (bahasa).

BIAYA YANG WAJIB MASUK NASKAH, bukan disembunyikan:
  - indexing menuntut JARINGAN dan kuota API; korpus tidak lagi dapat diindeks luring
  - menambah ketergantungan layanan pihak ketiga pada jalur inti sistem
  - vektor 3072 dimensi memperbesar indeks kira-kira 8x dibanding 384

CATATAN CACAT. config.yaml:119 menuliskan "models/embedding-001", yang per
16 Agustus 2026 SUDAH TIDAK ADA pada API Google (diperiksa lewat models.list,
tidak muncul di antara 53 model). Skrip ini menimpanya lewat env GOOGLE_EMBED_MODEL
dan TIDAK menyentuh config.yaml; penggantian nilai di config menunggu keputusan.

VARIAN (chunk 900/120, pemisah lama — identik, sehingga label tidak bergeser)
  EN256  all-MiniLM-L6-v2 lokal, jendela 256   -> kontrol
  GEM2   models/gemini-embedding-2 via API     -> kandidat produksi

DUGAAN
  H1 GEM2 nol pemotongan (jendela 8192 >> potongan terpanjang 443 token).
  H2 Dimensi GEM2 3072, berbeda dari 384. Dicatat supaya tidak ada yang menyimpulkan
     perbandingan ini mengisolasi satu faktor.
  H3 Arah MRR TIDAK didugakan. Kriteria di muka: selisih > 0,02 dianggap perbaikan
     nyata; |selisih| < 0,02 berarti biaya ketergantungan jaringan TIDAK terbayar.

Keluaran: results/eval_rag/embedding_gemini.json
"""
from __future__ import annotations

import torch  # noqa: F401
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ablation_rag_fullkb import TOP_K  # noqa: E402
from src.rag.ablation_query import build_ablation_query  # noqa: E402
from tuning_protocol import KRITERIA, bagi_dua, catat  # noqa: E402

CORPUS_TAG = os.environ.get("CORPUS_TAG", "kb12_sym")
OUT = ROOT / "results/eval_rag/embedding_gemini.json"
TMP = ROOT / "models/_embedding_gemini"
MODEL_GEMINI = os.environ.get("GEMINI_EMBED_UJI", "models/gemini-embedding-2")

# (kode, provider, model, jendela_efektif, keterangan)
VARIAN = [
    ("EN256", "sentence-transformers", "all-MiniLM-L6-v2", 256, "kontrol — lokal, luring"),
    ("GEM2", "google", MODEL_GEMINI, 8192, "kandidat produksi — terkelola, daring"),
]


def _muat_kunci() -> None:
    p = ROOT / ".env"
    if not p.exists():
        return
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if s.startswith("GOOGLE_API_KEY=") and not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = s.split("=", 1)[1].strip().strip('"')


def muat_skenario() -> dict:
    out = {}
    for berkas, nama in (("per_case_divergen.csv", "divergen"),
                         ("per_case_natural.csv", "natural")):
        p = ROOT / f"results/retrieval_realcases_{CORPUS_TAG}/{berkas}"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("mode") == "pc_rag"]
        out[f"{nama}_prediction_conditioned"] = [
            (build_ablation_query(float(r["query_glucose"])), r["expected"]) for r in rows]
        out[f"{nama}_standard"] = [
            (build_ablation_query(float(r["current"])), r["expected"]) for r in rows]
    return out


def env_varian(provider: str, model: str) -> dict:
    e = dict(os.environ, PYTHONPATH=str(ROOT), EMBED_PROVIDER=provider)
    if provider == "google":
        e["GOOGLE_EMBED_MODEL"] = model
    else:
        e["HF_EMBED_MODEL"] = model
        e["EMBED_MAX_SEQ_LENGTH"] = "256"
    return e


def indeks_ulang(provider: str, model: str, persist: Path) -> int:
    if persist.exists():
        shutil.rmtree(persist, ignore_errors=True)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/reingest_kb.py"),
         "--persist", str(persist), "--collection", "diabetes_kb",
         "--chunk-size", "900", "--chunk-overlap", "120", "--embed", provider],
        cwd=str(ROOT), env=env_varian(provider, model), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"reingest gagal ({provider}/{model}):\n"
                           f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    import chromadb
    n = chromadb.PersistentClient(path=str(persist)).get_collection("diabetes_kb").count()
    if n == 0:
        raise RuntimeError("Indeks kosong — hasil tidak sah.")
    return n


def evaluasi(pasangan, persist: Path, provider: str, model: str) -> dict:
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "_pasangan.json").write_text(json.dumps(pasangan), encoding="utf-8")
    kode = (
        "import sys,json,warnings;warnings.filterwarnings('ignore');"
        "sys.path.insert(0,r'%s');sys.path.insert(0,r'%s');"
        "from ablation_rag_fullkb import classify_chunk,TOP_K;"
        "from src.rag.retriever import MMRRetriever;"
        "pas=json.load(open(r'%s',encoding='utf-8'));"
        "r=MMRRetriever(persist_dir=r'%s',collection_name='diabetes_kb',embed_provider='%s');"
        "ranks=[]\n"
        "for q,e in pas:\n"
        "    t=[classify_chunk(d['text']) for d in r.retrieve(q,top_k=TOP_K)]\n"
        "    ranks.append((t.index(e)+1) if e in t else 0)\n"
        "print('JSON'+json.dumps(ranks))"
    ) % (ROOT, ROOT / "scripts", TMP / "_pasangan.json", persist, provider)
    proc = subprocess.run([sys.executable, "-c", kode], cwd=str(ROOT),
                          env=env_varian(provider, model), capture_output=True, text=True)
    baris = [l for l in proc.stdout.splitlines() if l.startswith("JSON")]
    if not baris:
        raise RuntimeError(f"evaluasi gagal:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    ranks = json.loads(baris[-1][4:])
    n = max(len(ranks), 1)
    return {
        "n": len(ranks),
        "hit@1(%)": round(100 * sum(1 for x in ranks if x == 1) / n, 1),
        f"hit@{TOP_K}(%)": round(100 * sum(1 for x in ranks if 0 < x <= TOP_K) / n, 1),
        "mrr": round(sum((1.0 / x) if x else 0.0 for x in ranks) / n, 3),
    }


def main() -> None:
    _muat_kunci()
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY tidak ada — jalur google tidak dapat diuji.")
    TMP.mkdir(parents=True, exist_ok=True)

    skenario = muat_skenario()
    if not skenario:
        raise SystemExit(f"Skenario tidak ditemukan untuk CORPUS_TAG={CORPUS_TAG}")
    setel, lapor = [], []
    for _, pasangan in skenario.items():
        b = bagi_dua(pasangan)
        if b:
            setel += b[0]
            lapor += b[1]
    print(f"Set penyetelan n={len(setel)} | set pelaporan n={len(lapor)} (tidak beririsan)")
    print(f"Kriteria ditetapkan di muka: {KRITERIA}")
    print(f"Model Gemini yang diuji: {MODEL_GEMINI}\n")

    hs, hl, wk, dim, jml = {}, {}, {}, {}, {}
    for kode, provider, model, jendela, ket in VARIAN:
        t0 = time.time()
        persist = TMP / kode.lower()
        jml[kode] = indeks_ulang(provider, model, persist)
        import chromadb
        dim[kode] = len(chromadb.PersistentClient(path=str(persist)).get_collection(
            "diabetes_kb").get(limit=1, include=["embeddings"])["embeddings"][0])
        hs[kode] = evaluasi(setel, persist, provider, model)
        hl[kode] = evaluasi(lapor, persist, provider, model)
        wk[kode] = round(time.time() - t0, 1)
        catat("embedding_gemini", kode, {**hs[kode], "n_chunk": jml[kode], "dim": dim[kode]},
              catatan=f"{provider}/{model}, jendela {jendela}, {ket}, korpus={CORPUS_TAG}")
        print(f"  {kode:6} {model:32} {jml[kode]:5} chunk dim {dim[kode]:5} | "
              f"MRR lapor {hl[kode]['mrr']:.3f} hit@1 {hl[kode]['hit@1(%)']:5.1f}% "
              f"({wk[kode]:.0f} dtk)", flush=True)

    d = round(hl["GEM2"]["mrr"] - hl["EN256"]["mrr"], 3)
    if d > 0.02:
        putusan = (f"gemini-embedding-2 mengungguli MiniLM lokal sebesar {d:+.3f}. "
                   "Perbaikan nyata, TETAPI percobaan ini mengubah empat faktor sekaligus "
                   "sehingga TIDAK menunjukkan faktor mana yang bekerja. Biaya "
                   "ketergantungan jaringan wajib disebut di Bab keterbatasan.")
    elif d < -0.02:
        putusan = (f"gemini-embedding-2 KALAH sebesar {d:+.3f}. Tidak diadopsi; "
                   "ketergantungan jaringan tidak terbayar sama sekali.")
    else:
        putusan = (f"Selisih {d:+.3f} berada di bawah ambang 0,02. Embedding terkelola "
                   "TIDAK memberi perbaikan yang sepadan dengan biaya ketergantungan "
                   "jaringan dan kuota. Temuan ini dilaporkan apa adanya.")

    hasil = {
        "percobaan": "T10 — embedding Gemini terkelola lawan MiniLM lokal",
        "prapendaftaran": "docstring scripts/eval_embedding_gemini.py",
        "PERINGATAN_TAFSIR": (
            "Percobaan ini mengubah EMPAT faktor sekaligus (bahasa, jendela, dimensi, "
            "keluarga model) dan karena itu TIDAK dapat mengisolasi penyebab. Ia "
            "menjawab pertanyaan praktis kelayakan, bukan pertanyaan mekanisme. "
            "Pemisahan penyebab ada pada T8 (jendela) dan T9 (bahasa)."),
        "biaya_yang_wajib_dilaporkan": [
            "indexing menuntut jaringan dan kuota API; korpus tidak lagi dapat diindeks luring",
            "menambah ketergantungan layanan pihak ketiga pada jalur inti sistem",
            "vektor 3072 dimensi memperbesar indeks kira-kira 8x dibanding 384",
        ],
        "cacat_config": ("config.yaml:119 menuliskan models/embedding-001 yang sudah "
                         "tidak ada pada API per 16 Agustus 2026; ditimpa lewat env "
                         "GOOGLE_EMBED_MODEL, config TIDAK disentuh"),
        "kriteria": KRITERIA,
        "n_set_penyetelan": len(setel), "n_set_pelaporan": len(lapor),
        "varian": {k: {"provider": p, "model": m, "jendela": j, "keterangan": ket}
                   for k, p, m, j, ket in VARIAN},
        "n_chunk": jml, "dimensi_vektor": dim, "waktu_dtk": wk,
        "hasil_set_penyetelan": hs, "set_pelaporan": hl,
        "dugaan": {
            "H1_gemini_nol_pemotongan": "jendela 8192 >> potongan terpanjang 443 token",
            "H2_dimensi_berbeda": dim,
            "H3_arah_MRR": "tidak didugakan; ambang 0,02 ditetapkan di muka",
        },
        "delta_mrr_GEM2_minus_EN256": d,
        "putusan": putusan,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== SET PELAPORAN (n={len(lapor)}) ===")
    for k in hl:
        print(f"  {k:6}: MRR {hl[k]['mrr']:.3f}  hit@1 {hl[k]['hit@1(%)']:5.1f}%  dim {dim[k]}")
    print(f"\ndelta MRR GEM2-EN256 = {d:+.3f}\n{putusan}")
    print(f"\nDisimpan ke {OUT}")


if __name__ == "__main__":
    main()
