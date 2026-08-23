"""T9 — apakah KETIDAKCOCOKAN BAHASA yang menjadi langit-langit retrieval?

PRAPENDAFTARAN, ditulis SEBELUM hasil retrieval varian mana pun dilihat.

LATAR. T8 menunjukkan pemotongan senyap BUKAN penyebab langit-langit MRR ~0,5:
menghapusnya (jendela 256 -> 512, potongan identik) justru menurunkan MRR 0,023
(results/eval_rag/jendela_token.json). Karena itu penyebabnya harus dicari di
tempat lain.

DUGAAN YANG DIUJI. Model produksi `all-MiniLM-L6-v2` adalah model BAHASA INGGRIS,
sedangkan korpusnya berbahasa Indonesia (PERKENI, IDAI, ISPAD). Ruang semantiknya
untuk bahasa Indonesia praktis kebetulan, bukan hasil pelatihan.

MENGAPA PERCOBAAN INI SAH. Seluruh varian memakai chunk_size, chunk_overlap, dan
daftar pemisah yang PERSIS SAMA, sehingga potongannya identik teks demi teks dan
label classify_chunk() tidak bergeser. Konfound yang membatalkan perbandingan T7
tidak berlaku. Yang berubah hanya model embedding.

RANCANGAN YANG MEMBUATNYA MENENTUKAN. Model multibahasa yang diuji ber-jendela
128 — LEBIH KECIL daripada 256 milik model Inggris, sehingga ia terpotong LEBIH
PARAH. Bila model multibahasa tetap menang meskipun dirugikan pada sumbu yang
selama ini disangka penyebabnya, maka bahasa terbukti lebih menentukan daripada
jendela. Rancangan ini sengaja memberatkan pihak yang didugakan menang.

VARIAN (chunk 900/120, pemisah lama, identik di semua varian)
  EN256  all-MiniLM-L6-v2                        jendela 256  -> kontrol (= W256)
  ML128  paraphrase-multilingual-MiniLM-L12-v2   jendela 128  -> jendela bawaannya
  ML256  paraphrase-multilingual-MiniLM-L12-v2   jendela 256  -> jendela disamakan

DUGAAN
  H1 ML128 membuang LEBIH BANYAK token daripada EN256 (jendela lebih kecil).
     Bila tidak, ada yang salah pada penyetelan varian.
  H2 Arah MRR TIDAK didugakan. Kriteria ditetapkan di muka: bila MRR multibahasa
     terbaik melampaui EN256 lebih dari 0,02 maka dugaan ketidakcocokan bahasa
     DIDUKUNG; bila |selisih| < 0,02 maka TIDAK didukung dan penyebab langit-langit
     harus dicari di tempat ketiga; bila kalah lebih dari 0,02 maka dugaan ini
     GUGUR dan dilaporkan gugur.
  H3 Kedua model berdimensi 384, sehingga perbedaan hasil TIDAK dapat dijelaskan
     oleh perbedaan dimensi vektor. Diperiksa, bukan diandaikan.

Keluaran: results/eval_rag/model_embedding.json
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
OUT = ROOT / "results/eval_rag/model_embedding.json"
TMP = ROOT / "models/_model_embedding"

EN = "all-MiniLM-L6-v2"
ML = "paraphrase-multilingual-MiniLM-L12-v2"

# (kode, model, jendela, keterangan)
VARIAN = [
    ("EN256", EN, 256, "kontrol — model Inggris, jendela bawaannya"),
    ("ML128", ML, 128, "multibahasa, jendela bawaannya (LEBIH KECIL — dirugikan)"),
    ("ML256", ML, 256, "multibahasa, jendela disamakan dengan kontrol"),
]


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


def env_varian(model: str, jendela: int) -> dict:
    """Environment yang MENGUNCI model dan jendela untuk satu varian.

    Lewat environment supaya proses indexing dan proses evaluasi memakai nilai
    yang sama. Model embedding yang berbeda antara ingest dan query membuat
    retrieval merosot menjadi derau TANPA error apa pun.
    """
    return dict(os.environ, PYTHONPATH=str(ROOT),
                HF_EMBED_MODEL=model, EMBED_MAX_SEQ_LENGTH=str(jendela))


def indeks_ulang(model: str, jendela: int, persist: Path) -> int:
    if persist.exists():
        shutil.rmtree(persist, ignore_errors=True)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/reingest_kb.py"),
         "--persist", str(persist), "--collection", "diabetes_kb",
         "--chunk-size", "900", "--chunk-overlap", "120"],
        cwd=str(ROOT), env=env_varian(model, jendela), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"reingest gagal ({model}@{jendela}):\n"
                           f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    import chromadb
    n = chromadb.PersistentClient(path=str(persist)).get_collection("diabetes_kb").count()
    if n == 0:
        raise RuntimeError("Indeks kosong — hasil tidak sah.")
    return n


def evaluasi(pasangan, persist: Path, model: str, jendela: int) -> dict:
    """Evaluasi dalam SUBPROSES: RagConfig di-cache lru_cache per proses."""
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "_pasangan.json").write_text(json.dumps(pasangan), encoding="utf-8")
    kode = (
        "import sys,json,warnings;warnings.filterwarnings('ignore');"
        "sys.path.insert(0,r'%s');sys.path.insert(0,r'%s');"
        "from ablation_rag_fullkb import classify_chunk,TOP_K;"
        "from src.rag.retriever import MMRRetriever;"
        "pas=json.load(open(r'%s',encoding='utf-8'));"
        "r=MMRRetriever(persist_dir=r'%s',collection_name='diabetes_kb',embed_provider='sentence-transformers');"
        "ranks=[]\n"
        "for q,e in pas:\n"
        "    t=[classify_chunk(d['text']) for d in r.retrieve(q,top_k=TOP_K)]\n"
        "    ranks.append((t.index(e)+1) if e in t else 0)\n"
        "print('JSON'+json.dumps(ranks))"
    ) % (ROOT, ROOT / "scripts", TMP / "_pasangan.json", persist)
    proc = subprocess.run([sys.executable, "-c", kode], cwd=str(ROOT),
                          env=env_varian(model, jendela), capture_output=True, text=True)
    baris = [l for l in proc.stdout.splitlines() if l.startswith("JSON")]
    if not baris:
        raise RuntimeError(f"evaluasi gagal:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    ranks = json.loads(baris[-1][4:])
    n = max(len(ranks), 1)
    return {
        "n": len(ranks),
        "hit@1(%)": round(100 * sum(1 for x in ranks if x == 1) / n, 1),
        f"hit@{TOP_K}(%)": round(100 * sum(1 for x in ranks if 0 < x <= TOP_K) / n, 1),
        "mrr": round(sum((1.0 / x) if x else 0.0 for x in ranks) / n, 3),
    }


def sifat(persist: Path, model: str, jendela: int) -> dict:
    import chromadb
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"sentence-transformers/{model}")
    teks = chromadb.PersistentClient(path=str(persist)).get_collection(
        "diabetes_kb").get(include=["documents"])["documents"]
    pj = [len(tok.encode(t)) for t in teks]
    n = max(len(pj), 1)
    return {
        "n_chunk": len(teks), "jendela": jendela,
        "token_rerata": round(sum(pj) / n, 1), "token_maks": max(pj),
        "persen_melewati_jendela": round(100 * sum(1 for x in pj if x > jendela) / n, 1),
        "persen_token_terbuang": round(
            100 * sum(max(0, x - jendela) for x in pj) / max(sum(pj), 1), 2),
    }


def main() -> None:
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
    print(f"Kriteria ditetapkan di muka: {KRITERIA}\n")

    hs, hl, sf, wk, dim = {}, {}, {}, {}, {}
    for kode, model, jendela, ket in VARIAN:
        t0 = time.time()
        persist = TMP / kode.lower()
        n = indeks_ulang(model, jendela, persist)
        sf[kode] = sifat(persist, model, jendela)
        import chromadb
        dim[kode] = chromadb.PersistentClient(path=str(persist)).get_collection(
            "diabetes_kb").get(limit=1, include=["embeddings"])["embeddings"][0].__len__()
        hs[kode] = evaluasi(setel, persist, model, jendela)
        hl[kode] = evaluasi(lapor, persist, model, jendela)
        wk[kode] = round(time.time() - t0, 1)
        catat("model_embedding", kode, {**hs[kode], **sf[kode]},
              catatan=f"{model} @ {jendela}, {ket}, korpus={CORPUS_TAG}")
        print(f"  {kode:6} {model:42} j={jendela:3} {n:5} chunk | "
              f"MRR lapor {hl[kode]['mrr']:.3f} hit@1 {hl[kode]['hit@1(%)']:5.1f}% | "
              f"terbuang {sf[kode]['persen_token_terbuang']:5.2f}% ({wk[kode]:.0f} dtk)", flush=True)

    ml_terbaik = max(["ML128", "ML256"], key=lambda k: hl[k]["mrr"])
    d = round(hl[ml_terbaik]["mrr"] - hl["EN256"]["mrr"], 3)
    h1 = sf["ML128"]["persen_token_terbuang"] > sf["EN256"]["persen_token_terbuang"]
    h3 = len(set(dim.values())) == 1

    if d > 0.02:
        putusan = (f"Dugaan ketidakcocokan bahasa DIDUKUNG: {ml_terbaik} mengungguli model "
                   f"Inggris sebesar {d:+.3f} pada set pelaporan, meskipun ML128 terpotong "
                   f"lebih parah. Bahasa lebih menentukan daripada jendela token.")
    elif d < -0.02:
        putusan = ("Dugaan ketidakcocokan bahasa GUGUR: model multibahasa kalah nyata. "
                   "Dilaporkan gugur; penyebab langit-langit harus dicari di tempat lain.")
    else:
        putusan = ("Dugaan ketidakcocokan bahasa TIDAK didukung (selisih < 0,02). "
                   "Penyebab langit-langit MRR bukan bahasa dan bukan pemotongan — "
                   "kandidat berikutnya adalah alat ukur dan susunan kueri itu sendiri.")

    hasil = {
        "percobaan": "T9 — pengaruh kecocokan bahasa model embedding terhadap retrieval",
        "prapendaftaran": "docstring scripts/eval_model_embedding.py, ditulis sebelum hasil dilihat",
        "mengapa_sah": ("Seluruh varian berpotongan IDENTIK (chunk 900/120, pemisah sama), "
                        "sehingga label classify_chunk() tidak bergeser. Yang berubah hanya "
                        "model embedding."),
        "rancangan_memberatkan_pihak_yang_didugakan_menang": (
            "ML128 berjendela 128, LEBIH KECIL daripada 256 milik kontrol, sehingga ia "
            "terpotong lebih parah pada sumbu yang selama ini disangka penyebabnya."),
        "kriteria": KRITERIA,
        "n_set_penyetelan": len(setel), "n_set_pelaporan": len(lapor),
        "varian": {k: {"model": m, "jendela": j, "keterangan": ket}
                   for k, m, j, ket in VARIAN},
        "integritas": sf,
        "dimensi_vektor": dim,
        "waktu_dtk": wk,
        "hasil_set_penyetelan": hs,
        "set_pelaporan": hl,
        "dugaan": {
            "H1_ML128_terbuang_lebih_banyak": h1,
            "H3_dimensi_sama_sehingga_bukan_penjelasnya": h3,
            "H2_arah_MRR": "tidak didugakan; ambang 0,02 ditetapkan di muka",
        },
        "model_multibahasa_terbaik": ml_terbaik,
        "delta_mrr_multibahasa_minus_inggris": d,
        "putusan": putusan,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== SET PELAPORAN (n={len(lapor)}) ===")
    for k in hl:
        print(f"  {k:6}: MRR {hl[k]['mrr']:.3f}  hit@1 {hl[k]['hit@1(%)']:5.1f}%  "
              f"terbuang {sf[k]['persen_token_terbuang']:5.2f}%  dim {dim[k]}")
    print(f"\nH1 ML128 terbuang lebih banyak : {'LOLOS' if h1 else 'GAGAL'} "
          f"({sf['ML128']['persen_token_terbuang']}% vs {sf['EN256']['persen_token_terbuang']}%)")
    print(f"H3 dimensi sama (bukan penjelas): {'LOLOS' if h3 else 'GAGAL'} {dim}")
    print(f"\ndelta MRR {ml_terbaik} - EN256 = {d:+.3f}\n{putusan}")
    print(f"\nDisimpan ke {OUT}")


if __name__ == "__main__":
    main()
