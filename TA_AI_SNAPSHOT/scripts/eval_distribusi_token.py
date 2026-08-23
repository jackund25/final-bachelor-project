"""T5.1 — distribusi panjang token per chunk, untuk chunk_size produksi (900) dan
pembanding (500).

MENGAPA PANJANG TOKEN, BUKAN PANJANG KARAKTER
---------------------------------------------
`chunk_size` di `config.yaml` dinyatakan dalam KARAKTER, sedangkan yang membatasi model
embedding adalah TOKEN. Keduanya tidak proporsional dan nisbahnya berbeda antar-dokumen:
teks bertabel dan bersingkatan medis menghasilkan lebih banyak token per karakter daripada
prosa biasa.

Yang membuat selisih itu penting: `all-MiniLM-L6-v2` memiliki
``max_seq_length`` = **256 token** (dibaca langsung dari `sentence_bert_config.json` model
yang terpasang, bukan diasumsikan). Token ke-257 dan seterusnya **dipotong secara diam-diam**
— tanpa peringatan, tanpa error. Isi di ekor chunk yang terpotong tidak pernah masuk ke
vektor, sehingga tidak akan pernah terambil retrieval betapa pun relevannya.

Dengan kata lain, chunk_size yang terlalu besar tidak sekadar "kurang optimal": ia dapat
membuat sebagian korpus **tidak terindeks sama sekali** sementara `manifest.csv` dan jumlah
chunk tetap terlihat wajar. Skrip ini mengukur berapa besar bagian itu.

Perhatikan bahwa `model_max_length` yang dilaporkan tokenizer adalah **512** — angka dari
konfigurasi BERT-nya, bukan batas yang benar-benar dipakai sentence-transformers. Memakai
512 akan membuat masalahnya tampak tidak ada. Skrip ini memakai 256.

Keluaran: results/eval_rag/distribusi_token.json
"""
from __future__ import annotations

import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "results/eval_rag"
# Produksi (900) dan pembanding (500) yang diminta T5.1, DITAMBAH tiga ukuran lain yang
# sudah pernah disapu B4. Dengan kelimanya, proporsi pemotongan dapat disandingkan
# terhadap MRR yang sudah terukur pada sapuan itu — lima titik memperlihatkan hubungan,
# sedangkan dua titik hanya memperlihatkan selisih.
UKURAN_DIUJI = [900, 500, 300, 1400, 2000]
# MRR set penyetelan dari sapuan B4 (results/tuning_log.json, korpus kb12_sym, n=240).
# Disalin ke sini HANYA untuk penyandingan; sumber kebenarannya tetap tuning_log.json.
MRR_B4 = {300: 0.306, 500: 0.541, 900: 0.425, 1400: 0.313, 2000: 0.302}
HIT1_B4 = {300: 3.3, 500: 40.0, 900: 26.2, 1400: 16.2, 2000: 13.8}
PERSENTIL = [5, 25, 50, 75, 90, 95, 99]


def batas_token_model(nama_model: str) -> tuple[int, str, str]:
    """Batas panjang urutan yang BENAR-BENAR dipakai sentence-transformers.

    Dibaca dari `sentence_bert_config.json` di snapshot model terpasang. Kalau berkas itu
    tidak ditemukan, skrip BERHENTI alih-alih menebak: menebak 512 (nilai konfigurasi BERT)
    akan menyembunyikan tepat masalah yang sedang diukur.

    `config.yaml` menulis nama pendek (`all-MiniLM-L6-v2`) sedangkan cache HuggingFace
    memakai nama berorganisasi (`sentence-transformers/all-MiniLM-L6-v2`), jadi keduanya
    dicoba. Bila nama pendek cocok dengan LEBIH DARI SATU organisasi, skrip berhenti:
    dua organisasi dapat memakai max_seq_length berbeda, dan memilih salah satunya
    diam-diam akan menghasilkan batas yang keliru tanpa ada yang gagal.
    """
    hub = Path.home() / ".cache/huggingface/hub"
    pendek = nama_model.split("/")[-1]
    kandidat = sorted(hub.glob(f"models--*--{pendek}/snapshots/*/sentence_bert_config.json")) \
        + sorted(hub.glob(f"models--{pendek}/snapshots/*/sentence_bert_config.json"))
    if not kandidat:
        raise SystemExit(
            f"sentence_bert_config.json untuk {nama_model} tidak ditemukan di cache "
            f"({hub}). Batas token tidak boleh ditebak — jalankan ulang setelah model "
            f"terunduh.")
    repo = {p.parents[2].name for p in kandidat}
    if len(repo) > 1:
        raise SystemExit(
            f"Nama '{pendek}' cocok dengan lebih dari satu repo di cache: {sorted(repo)}. "
            f"Batas token dapat berbeda antar-repo; tuliskan nama lengkap berorganisasi "
            f"pada rag.embedding_model agar tidak ambigu.")
    cfg = json.loads(kandidat[0].read_text(encoding="utf-8"))
    nama_penuh = kandidat[0].parents[2].name.replace("models--", "").replace("--", "/")
    return int(cfg["max_seq_length"]), str(kandidat[0]), nama_penuh


def ringkas(panjang: list[int], batas: int) -> dict:
    a = np.asarray(panjang, dtype=float)
    lewat = a > batas
    # Berapa banyak token yang benar-benar hilang, bukan sekadar berapa chunk yang lewat.
    hilang = np.clip(a - batas, 0, None)
    return {
        "n_chunk": int(len(a)),
        "token_min": int(a.min()), "token_maks": int(a.max()),
        "token_rerata": round(float(a.mean()), 1),
        "persentil": {str(p): round(float(np.percentile(a, p)), 1) for p in PERSENTIL},
        "n_melewati_batas": int(lewat.sum()),
        "persen_melewati_batas": round(100 * float(lewat.mean()), 2),
        "total_token": int(a.sum()),
        "total_token_terpotong": int(hilang.sum()),
        "persen_token_terpotong": round(100 * float(hilang.sum() / a.sum()), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-page-chars", type=int, default=200)
    args = ap.parse_args()

    from src.config import load_rag_config
    from src.rag.knowledge_base import MedicalKnowledgeBase
    from reingest_kb import _build_page_docs, _load_manifest, _validate_corpus
    from transformers import AutoTokenizer

    rag_cfg = load_rag_config()
    nama_model = rag_cfg.embedding_model
    batas, sumber_batas, nama_penuh = batas_token_model(nama_model)
    tok = AutoTokenizer.from_pretrained(nama_penuh)

    print(f"T5.1 — distribusi token per chunk")
    print(f"model embedding : {nama_model}  (di cache: {nama_penuh})")
    print(f"batas token     : {batas} (max_seq_length, dari {Path(sumber_batas).parent.name})")
    print(f"model_max_length tokenizer : {tok.model_max_length} "
          f"<- BUKAN batas yang dipakai; memakai angka ini akan menyembunyikan masalahnya")
    print(f"chunk_size produksi (config): {rag_cfg.chunk_size}, "
          f"overlap {rag_cfg.chunk_overlap}\n")

    pdf_dir = ROOT / "data/knowledge_base/books"
    manifest = _load_manifest(ROOT / "data/knowledge_base/manifest.csv")
    pasangan = _validate_corpus(pdf_dir, manifest)

    docs = []
    for pdf_path, entry in pasangan:
        page_docs, _, _, _ = _build_page_docs(pdf_path, entry, args.min_page_chars)
        docs.extend(page_docs)
    print(f"korpus: {len(docs)} halaman dari {len(pasangan)} dokumen\n")

    kb = MedicalKnowledgeBase()
    hasil = {}
    for ukuran in UKURAN_DIUJI:
        # Overlap diskalakan sebanding supaya perbandingannya adil: overlap tetap pada
        # chunk yang lebih kecil berarti proporsi tumpang tindih yang jauh lebih besar,
        # dan selisih yang terukur menjadi campuran dua perubahan.
        overlap = int(round(rag_cfg.chunk_overlap * ukuran / rag_cfg.chunk_size))
        chunks = kb.chunk_documents(documents=docs, chunk_size=ukuran,
                                    chunk_overlap=overlap)
        teks = [c["text"] for c in chunks]
        panjang = [len(tok.encode(t, add_special_tokens=True, truncation=False))
                   for t in teks]
        karakter = [len(t) for t in teks]

        r = ringkas(panjang, batas)
        r.update({
            "chunk_size_karakter": ukuran, "chunk_overlap_karakter": overlap,
            "adalah_produksi": ukuran == rag_cfg.chunk_size,
            "karakter_rerata": round(float(np.mean(karakter)), 1),
            "karakter_per_token": round(float(np.sum(karakter) / np.sum(panjang)), 3),
        })

        # Per dokumen: nisbah karakter-per-token berbeda antar-dokumen, jadi dokumen
        # tertentu bisa jauh lebih terdampak daripada rerata korpus.
        per_dok = defaultdict(list)
        for c, p in zip(chunks, panjang):
            per_dok[c.get("source") or c.get("metadata", {}).get("source", "?")].append(p)
        r["per_dokumen"] = {
            k: {"n_chunk": len(v), "token_median": round(float(np.median(v)), 1),
                "token_maks": int(max(v)),
                "n_melewati_batas": int(sum(1 for x in v if x > batas)),
                "persen_melewati_batas": round(100 * sum(1 for x in v if x > batas) / len(v), 2)}
            for k, v in sorted(per_dok.items())
        }
        hasil[str(ukuran)] = r

        tag = " (PRODUKSI)" if r["adalah_produksi"] else " (pembanding)"
        print(f"chunk_size={ukuran} overlap={overlap}{tag}")
        print(f"  {r['n_chunk']:,} chunk | token rerata {r['token_rerata']} "
              f"| median {r['persentil']['50']} | p95 {r['persentil']['95']} "
              f"| maks {r['token_maks']}")
        print(f"  karakter/token {r['karakter_per_token']}")
        print(f"  MELEWATI BATAS {batas}: {r['n_melewati_batas']:,} chunk "
              f"({r['persen_melewati_batas']}%) | "
              f"token terpotong {r['total_token_terpotong']:,} "
              f"({r['persen_token_terpotong']}% dari seluruh token)")
        terparah = sorted(r["per_dokumen"].items(),
                          key=lambda kv: -kv[1]["persen_melewati_batas"])[:3]
        for nama, v in terparah:
            if v["persen_melewati_batas"] > 0:
                print(f"    terdampak: {nama} — {v['persen_melewati_batas']}% chunk "
                      f"({v['n_melewati_batas']}/{v['n_chunk']})")
        print()

    # Penyandingan terhadap MRR B4. Korelasi peringkat dipakai, bukan Pearson: yang
    # ditanyakan adalah apakah URUTANNYA sejalan, bukan apakah hubungannya linear.
    ukuran_urut = [u for u in UKURAN_DIUJI if u in MRR_B4]
    pot = [hasil[str(u)]["persen_token_terpotong"] for u in ukuran_urut]
    mrr = [MRR_B4[u] for u in ukuran_urut]
    try:
        rho, p_rho = stats.spearmanr(pot, mrr)
    except Exception:  # noqa: BLE001
        rho, p_rho = float("nan"), float("nan")

    print(f"{'chunk':>7}{'% chunk >batas':>16}{'% token hilang':>16}"
          f"{'MRR (B4)':>11}{'Hit@1 (B4)':>12}")
    for u in ukuran_urut:
        h = hasil[str(u)]
        print(f"{u:>7}{h['persen_melewati_batas']:>15.2f}%{h['persen_token_terpotong']:>15.2f}%"
              f"{MRR_B4[u]:>11.3f}{HIT1_B4[u]:>11.1f}%")
    print(f"\nSpearman(% token hilang, MRR) = {rho:.3f}  p={p_rho:.4f}  (n={len(ukuran_urut)})"
          f"  -> {'signifikan' if p_rho == p_rho and p_rho < 0.05 else 'TIDAK signifikan'}")
    print("Hubungannya TIDAK monoton: chunk 300 nol pemotongan tetapi MRR 0,306. "
          "Pemotongan bukan satu-satunya yang bekerja.")

    prod = hasil[str(rag_cfg.chunk_size)] if str(rag_cfg.chunk_size) in hasil else None
    out = {
        "percobaan": "T5.1 — distribusi panjang token per chunk",
        "model_embedding": nama_model,
        "model_embedding_penuh": nama_penuh,
        "batas_token_max_seq_length": batas,
        "sumber_batas": "sentence_bert_config.json pada snapshot model terpasang",
        "catatan_model_max_length": (
            f"tokenizer melaporkan model_max_length={tok.model_max_length}, tetapi itu "
            f"konfigurasi BERT-nya. sentence-transformers memotong pada {batas}. Memakai "
            f"512 akan membuat pemotongan tampak tidak pernah terjadi."),
        "chunk_size_produksi": rag_cfg.chunk_size,
        "chunk_overlap_produksi": rag_cfg.chunk_overlap,
        "catatan_overlap": ("Overlap diskalakan sebanding dengan chunk_size supaya "
                            "perbandingan 900 vs 500 tidak mencampur dua perubahan."),
        "n_halaman_korpus": len(docs),
        "hasil": hasil,
        "penyandingan_dengan_sapuan_B4": {
            "sumber_mrr": "results/tuning_log.json (parameter=chunk_size, korpus kb12_sym, n=240)",
            "per_ukuran": {str(u): {"persen_token_terpotong":
                                    hasil[str(u)]["persen_token_terpotong"],
                                    "persen_chunk_melewati_batas":
                                    hasil[str(u)]["persen_melewati_batas"],
                                    "mrr_B4": MRR_B4[u], "hit@1_B4": HIT1_B4[u]}
                           for u in ukuran_urut},
            "spearman_rho": None if rho != rho else round(float(rho), 4),
            "spearman_p": None if p_rho != p_rho else round(float(p_rho), 4),
            "HASIL_UJI": (
                "Spearman atas KELIMA ukuran TIDAK signifikan. Hubungannya tidak monoton, "
                "dan itu informatif: chunk 300 tidak terpotong sama sekali (0,00%) tetapi "
                "MRR-nya 0,306, terburuk kedua. Pemotongan karena itu JELAS bukan "
                "satu-satunya yang bekerja."),
            "penafsiran_dua_faktor": (
                "chunk_size menggerakkan dua hal yang berlawanan arah: makin besar chunk, "
                "makin banyak konteks per potongan (membantu), tetapi makin besar pula "
                "bagian yang dipotong diam-diam (merugikan). Optimum di sekitar 500 "
                "konsisten dengan kedua gaya itu bertemu, tetapi ini PENAFSIRAN atas pola, "
                "bukan hasil uji."),
            "pengamatan_post_hoc_JANGAN_dibaca_sebagai_uji": (
                "Di antara empat ukuran >=500, MRR turun monoton seiring naiknya proporsi "
                "pemotongan (0,01%->0,541; 8,23%->0,425; 29,66%->0,313; 47,95%->0,302), "
                "yaitu korelasi peringkat sempurna. Subhimpunan itu dipilih SETELAH melihat "
                "data, sehingga angka signifikansi apa pun atasnya tidak sah. Dicatat "
                "sebagai pengamatan yang mengarahkan percobaan berikutnya, bukan sebagai "
                "bukti."),
            "UJI_YANG_MEMISAHKAN_KEDUANYA": (
                "Pertahankan chunk_size 900 tetapi hilangkan pemotongannya — indeks ulang "
                "dengan model ber-max_seq_length lebih besar (mis. all-mpnet-base-v2) atau "
                "paksa setiap chunk <=256 token. Hanya satu faktor yang berubah, sehingga "
                "kenaikan MRR (bila ada) dapat dikaitkan pada pemotongan. Tanpa uji itu, "
                "pemotongan tetap berstatus mekanisme yang terukur tetapi belum terbukti "
                "sebagai penyebab."),
        },
        "RELEVANSI_BAGI_KEPUTUSAN_YANG_SUDAH_DIAMBIL": (
            "chunk_size dipertahankan 900 pada Tahap 0 dengan alasan selisih B4 tidak "
            "signifikan (p=0,078). Saat keputusan itu diambil, BELUM diketahui bahwa 900 "
            "membuang 8,23% token korpus secara diam-diam, dan bahwa KB-03 (dokumen sumber "
            "evaluasi RAGAS) kehilangan 48,84% chunk-nya. Ini informasi baru yang bersinggungan "
            "dengan keputusan tersebut, bukan pembatalannya."),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "distribusi_token.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    if prod is not None:
        if prod["n_melewati_batas"] == 0:
            print(f"KESIMPULAN: pada chunk_size produksi {rag_cfg.chunk_size}, TIDAK ADA "
                  f"chunk yang melewati batas {batas} token. Pemotongan diam-diam tidak "
                  f"terjadi, sehingga bukan penjelasan bagi lemahnya retrieval.")
        else:
            print(f"KESIMPULAN: pada chunk_size produksi {rag_cfg.chunk_size}, "
                  f"{prod['n_melewati_batas']:,} chunk ({prod['persen_melewati_batas']}%) "
                  f"melewati batas {batas} token; {prod['persen_token_terpotong']}% dari "
                  f"seluruh token korpus tidak pernah masuk ke vektor.")
    print(f"\nDisimpan ke {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
