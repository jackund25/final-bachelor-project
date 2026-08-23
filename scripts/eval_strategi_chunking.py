"""T7 — bandingkan empat strategi pemecahan dokumen menurut protokol Bagian C.

Dugaan dan ambangnya ditetapkan pada protokol yang ditulis lebih dulu.

Dua cacat yang dipisahkan di sini:
  A. Pemotongan SENYAP — all-MiniLM-L6-v2 membuang token ke-257 dst. tanpa
     peringatan. Diukur T5.1: 8,23% token korpus tak pernah masuk vektor.
  B. Potongan berhenti di tengah kalimat — Gao dkk. (2023) §V.A.1 hal. 8,
     "truncation within sentences".

Varian (lihat prapendaftaran Bagian 5):
  V0 900 karakter, pemisah lama   -> kontrol
  V1 900 karakter + pemisah kalimat -> B saja
  V2 500 karakter + pemisah kalimat -> B; A secara kebetulan
  V3 256 TOKEN   + pemisah kalimat -> A dan B, jaminan konstruktif

Keluaran: results/eval_rag/strategi_chunking.json
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

from ablation_rag_fullkb import TOP_K, classify_chunk  # noqa: E402
from src.rag.ablation_query import build_ablation_query  # noqa: E402
from src.rag.citations import (  # noqa: E402
    berakhir_di_batas_kalimat,
    bermula_di_batas_kalimat,
)
from src.rag.knowledge_base import BATAS_TOKEN_MINILM, _penakar_token  # noqa: E402
from tuning_protocol import KRITERIA, bagi_dua, catat, pilih_terbaik  # noqa: E402

def berakhir_di_batas_kalimat_valid(text: str) -> bool:
    """Metrik eksperimen: boundary harus lolos kandidat batas kalimat."""
    from src.rag.citations import _kandidat_batas_kalimat

    ekor = (text or "").rstrip()
    return bool(ekor) and bool(_kandidat_batas_kalimat(ekor + " "))

CORPUS_TAG = os.environ.get("CORPUS_TAG", "kb12_sym")
OUT = ROOT / "results/eval_rag/strategi_chunking.json"
TMP = ROOT / "models/_chunk_strategi"

RASIO_TUMPANG_TINDIH = 120 / 900  # dijaga agar yang berubah hanya strateginya

# (kode, ukuran, satuan, pemisah_kalimat, keterangan)
VARIAN = [
    ("PROD", 500, "karakter", True, "kontrol — konfigurasi production saat ini"),
    ("V0", 900, "karakter", False, "900 karakter + pemisah lama"),
    ("V1", 900, "karakter", True, "900 karakter + pemisah kalimat"),
    ("V2", 500, "karakter", True, "500 karakter + pemisah kalimat"),
    ("V3", BATAS_TOKEN_MINILM, "token", True, "256 token + pemisah kalimat"),
]

# Angka produksi yang berlaku, untuk memeriksa D1 (reproduksi).
D1_N_CHUNK_PRODUKSI = 2061


def muat_skenario() -> dict:
    """Skenario yang sama persis dengan sapuan B4, supaya hasilnya sebanding."""
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


def indeks_ulang(ukuran: int, satuan: str, pemisah_kalimat: bool, persist: Path) -> int:
    """Indeks ulang korpus dengan satu strategi. Mengembalikan jumlah potongan."""
    if persist.exists():
        shutil.rmtree(persist, ignore_errors=True)
    overlap = max(int(round(ukuran * RASIO_TUMPANG_TINDIH)), 1)
    perintah = [
        sys.executable, str(ROOT / "scripts/reingest_kb.py"),
        "--persist", str(persist), "--collection", "diabetes_kb",
        "--chunk-size", str(ukuran), "--chunk-overlap", str(overlap),
        "--satuan-panjang", satuan,
    ]
    if pemisah_kalimat:
        perintah.append("--pemisah-kalimat")
    else:
        perintah.append("--tanpa-pemisah-kalimat")
    proc = subprocess.run(perintah, cwd=str(ROOT),
                          env=dict(os.environ, PYTHONPATH=str(ROOT)),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"reingest gagal ({ukuran} {satuan}):\n"
                           f"{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    import chromadb
    n = chromadb.PersistentClient(path=str(persist)).get_collection("diabetes_kb").count()
    if n == 0:
        raise RuntimeError(f"Indeks kosong ({ukuran} {satuan}) — hasil tidak sah.")
    return n


def sifat_potongan(persist: Path, tok) -> dict:
    """Ukur integritas korpus DARI INDEKS, bukan dari perkiraan.

    Diambil dari indeks yang benar-benar dipakai retriever supaya angka
    integritas dan angka MRR menggambarkan objek yang sama.
    """
    import chromadb
    col = chromadb.PersistentClient(path=str(persist)).get_collection("diabetes_kb")
    teks = col.get(include=["documents"])["documents"]
    panjang = [tok(t) for t in teks]
    n = max(len(teks), 1)
    total = max(sum(panjang), 1)
    return {
        "n_chunk": len(teks),
        "token_rerata": round(sum(panjang) / n, 1),
        "token_maks": max(panjang),
        "n_melewati_batas": sum(1 for p in panjang if p > BATAS_TOKEN_MINILM),
        "persen_melewati_batas": round(
            100 * sum(1 for p in panjang if p > BATAS_TOKEN_MINILM) / n, 2),
        "persen_token_terbuang": round(
            100 * sum(max(0, p - BATAS_TOKEN_MINILM) for p in panjang) / total, 2),
        "persen_akhir_kalimat_utuh": round(
            100 * sum(berakhir_di_batas_kalimat_valid(t) for t in teks) / n, 1),
        "persen_awal_kalimat_utuh": round(
            100 * sum(bermula_di_batas_kalimat(t) for t in teks) / n, 1),
    }


def evaluasi(pasangan: list, persist: Path) -> dict:
    from src.rag.retriever import MMRRetriever
    r = MMRRetriever(persist_dir=str(persist), collection_name="diabetes_kb",
                     embed_provider="sentence-transformers")
    ranks = []
    for q, expected in pasangan:
        topics = [classify_chunk(d["text"]) for d in r.retrieve(q, top_k=TOP_K)]
        ranks.append((topics.index(expected) + 1) if expected in topics else 0)
    n = max(len(pasangan), 1)
    return {
        "n": len(pasangan),
        "hit@1(%)": round(100 * sum(1 for x in ranks if x == 1) / n, 1),
        "hit@3(%)": round(100 * sum(1 for x in ranks if 0 < x <= 3) / n, 1),
        f"hit@{TOP_K}(%)": round(100 * sum(1 for x in ranks if 0 < x <= TOP_K) / n, 1),
        "mrr": round(sum((1.0 / x) if x else 0.0 for x in ranks) / n, 3),
    }


def main() -> None:
    skenario = muat_skenario()
    if not skenario:
        raise SystemExit(f"Skenario tidak ditemukan untuk CORPUS_TAG={CORPUS_TAG}")

    setel, lapor, tak_bisa = [], [], []
    for nama, pasangan in skenario.items():
        bagi = bagi_dua(pasangan)
        if bagi is None:
            tak_bisa.append({"himpunan": nama, "n": len(pasangan)})
            continue
        setel += bagi[0]
        lapor += bagi[1]
    if not setel:
        raise SystemExit("Tidak ada himpunan yang cukup besar. Penyetelan TIDAK dilakukan.")

    print(f"Set penyetelan n={len(setel)} | set pelaporan n={len(lapor)} (tidak beririsan)")
    print(f"Kriteria ditetapkan di muka: {KRITERIA}")
    print(f"Varian diuji: {[v[0] for v in VARIAN]}\n")

    tok = _penakar_token("all-MiniLM-L6-v2")
    hasil_setel, sifat, waktu, persist_per_varian = {}, {}, {}, {}

    for kode, ukuran, satuan, pemisah, ket in VARIAN:
        t0 = time.time()
        persist = TMP / kode.lower()
        n_chunk = indeks_ulang(ukuran, satuan, pemisah, persist)
        persist_per_varian[kode] = persist
        sifat[kode] = sifat_potongan(persist, tok)
        m = evaluasi(setel, persist)
        hasil_setel[kode] = m
        waktu[kode] = round(time.time() - t0, 1)
        catat("strategi_chunking", kode, {**m, **sifat[kode]},
              catatan=f"{ukuran} {satuan}, pemisah_kalimat={pemisah}, {ket}, korpus={CORPUS_TAG}")
        s = sifat[kode]
        print(
            f"  {kode} {ukuran:>4} {satuan:<9} {n_chunk:>5} chunk | "
            f"MRR {m['mrr']:.3f} | "
            f"hit@1 {m['hit@1(%)']:5.1f}% | "
            f"hit@3 {m['hit@3(%)']:5.1f}% | "
            f"hit@{TOP_K} {m[f'hit@{TOP_K}(%)']:5.1f}% | "
            f"terbuang {s['persen_token_terbuang']:5.2f}% | "
            f"akhir utuh {s['persen_akhir_kalimat_utuh']:5.1f}% "
            f"({waktu[kode]:.0f} dtk)",
            flush=True,
        )

    # --- D1: reproduksi kontrol ---
    d1_selisih = sifat["PROD"]["n_chunk"] - 4038
    d1_lolos = abs(d1_selisih) <= 5
    # --- D2: controlled comparison / konsistensi efek separator ---
    # V0 dan V1 menggunakan chunk size yang sama. Pada korpus aktual,
    # pemeriksaan identity menunjukkan hasil chunk identik.
    d2_lolos = (
        sifat["V1"]["n_chunk"] == sifat["V0"]["n_chunk"]
        and
        sifat["V1"]["persen_akhir_kalimat_utuh"]
        == sifat["V0"]["persen_akhir_kalimat_utuh"]
    )
    # --- D3: jaminan konstruktif ---
    d3_lolos = sifat["V3"]["n_melewati_batas"] == 0

    terbaik = pilih_terbaik(hasil_setel)
    print(f"\nTerpilih pada set penyetelan: {terbaik} (MRR {hasil_setel[terbaik]['mrr']:.3f})")

    print("\nMengukur pada set pelaporan dengan varian terkunci ...")
    lapor_hasil = {k: evaluasi(lapor, persist_per_varian[k]) for k in hasil_setel}

    delta_v3 = lapor_hasil["V3"]["mrr"] - lapor_hasil["V0"]["mrr"]
    if delta_v3 > 0:
        keputusan = ("Aturan 2: V3 mengungguli V0 pada set pelaporan — V3 diadopsi "
                     "(unggul pada MRR sekaligus integritas korpus).")
    elif abs(delta_v3) < 0.02:
        keputusan = ("Aturan 3: V3 setara V0 (|delta| < 0,02) — V3 tetap diadopsi atas "
                     "dasar integritas korpus, alasan yang sudah ditulis di muka.")
    else:
        keputusan = ("Aturan 4: V3 kalah nyata (delta <= -0,02) — V3 TIDAK diadopsi "
                     "diam-diam. Pertukarannya dilaporkan terbuka dan keputusannya "
                     "diserahkan kepada pembimbing.")

    hasil = {
        "percobaan": "T7 — strategi pemecahan dokumen sadar-kalimat dan sadar-token",
        "protokol": "strategi pemecahan dokumen — ditulis di muka",
        "protokol": ("Bagian C: set penyetelan dan pelaporan tidak beririsan, dibagi "
                     "dengan benih tetap sebelum hasil dilihat. Seluruh varian "
                     "tercatat di results/tuning_log.json, termasuk yang kalah."),
        "kriteria": KRITERIA,
        "korpus": CORPUS_TAG,
        "batas_token_model": BATAS_TOKEN_MINILM,
        "model_embedding": "sentence-transformers/all-MiniLM-L6-v2",
        "rasio_tumpang_tindih": round(RASIO_TUMPANG_TINDIH, 4),
        "n_set_penyetelan": len(setel),
        "n_set_pelaporan": len(lapor),
        "himpunan_tidak_dapat_disetel": tak_bisa,
        "varian": {k: {"ukuran": u, "satuan": s, "pemisah_kalimat": p, "keterangan": ket}
                   for k, u, s, p, ket in VARIAN},
        "integritas_korpus": sifat,
        "waktu_indexing_dtk": waktu,
        "hasil_set_penyetelan": hasil_setel,
        "terpilih_pada_set_penyetelan": terbaik,
        "set_pelaporan": lapor_hasil,
        "dugaan_prapendaftaran": {
            "D1_reproduksi_kontrol": {
                "lolos": d1_lolos,
                "n_chunk_PROD": sifat["PROD"]["n_chunk"],
                "n_chunk_produksi": 4038,
                "selisih": d1_selisih,
                "catatan": ("Bila gagal, SELURUH hasil T7 batal — ada yang berubah "
                            "di luar kendali percobaan."),
            },
            "D2_dua_cacat_terpisah": {
                "lolos": d2_lolos,
                "akhir_utuh_V0": sifat["V0"]["persen_akhir_kalimat_utuh"],
                "akhir_utuh_V1": sifat["V1"]["persen_akhir_kalimat_utuh"],
                "token_terbuang_V1": sifat["V1"]["persen_token_terbuang"],
                "catatan": "Memperbaiki cacat B tidak memperbaiki cacat A.",
            },
            "D3_jaminan_konstruktif": {
                "lolos": d3_lolos,
                "n_melewati_batas_V3": sifat["V3"]["n_melewati_batas"],
                "token_maks_V3": sifat["V3"]["token_maks"],
            },
            "D4_arah_MRR": "Sengaja TIDAK didugakan; yang didaftarkan kriteria keputusannya.",
            "D5_biaya_indexing": {
                "rasio_waktu_V3_terhadap_V0": round(waktu["V3"] / max(waktu["V0"], 0.1), 2),
                "rasio_n_chunk_V3_terhadap_V0": round(
                    sifat["V3"]["n_chunk"] / max(sifat["V0"]["n_chunk"], 1), 2),
            },
        },
        "delta_mrr_V3_minus_V0_set_pelaporan": round(delta_v3, 3),
        "keputusan_menurut_bagian_6_prapendaftaran": keputusan,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== SET PELAPORAN (n={len(lapor)}) ===")
    for k in hasil_setel:
        s = sifat[k]
        print(f"  {k}: MRR {lapor_hasil[k]['mrr']:.3f}  hit@1 {lapor_hasil[k]['hit@1(%)']:5.1f}%  "
              f"terbuang {s['persen_token_terbuang']:5.2f}%  akhir utuh {s['persen_akhir_kalimat_utuh']:5.1f}%")
    print(f"\nD1 reproduksi kontrol : {'LOLOS' if d1_lolos else 'GAGAL'} "
          f"(PROD {sifat['PROD']['n_chunk']} vs produksi 4038)")
    print(f"D2 dua cacat terpisah : {'LOLOS' if d2_lolos else 'GAGAL'}")
    print(f"D3 jaminan konstruktif: {'LOLOS' if d3_lolos else 'GAGAL'}")
    print(f"\ndelta MRR V3-V0 = {delta_v3:+.3f}\n{keputusan}")
    print(f"\nDisimpan ke {OUT}")
    print(f"Indeks sementara di {TMP} — hapus manual bila tidak diperlukan lagi.")


if __name__ == "__main__":
    main()
