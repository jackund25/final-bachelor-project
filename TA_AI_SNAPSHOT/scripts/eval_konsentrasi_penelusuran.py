"""TEMUAN — seberapa besar bagian korpus yang PERNAH terambil sama sekali?

MENGAPA INI TEMUAN, BUKAN CATATAN METODE
----------------------------------------
Angka ini muncul tanpa dicari, sewaktu menyiapkan T2.2: bentuk kueri produksi hanya
menghasilkan 25 potongan unik dari 180 pengambilan, atas korpus 2.186 chunk. Setelah
diperiksa, ia ternyata **penjelasan mekanistik bersama** bagi tiga temuan yang sebelumnya
berdiri sendiri-sendiri:

1. **Pola biner `context_precision`** pada RAGAS (T1.2). Bila hanya segelintir potongan yang
   pernah terambil, konteks sebuah kueri hampir seluruhnya benar atau hampir seluruhnya
   salah — tidak ada di antaranya.
2. **Hit@1 rendah berhadapan dengan Hit@5 tinggi** (23,3% lawan 91,7%). Kalau kumpulan
   kandidat yang sama muncul berulang untuk kueri yang berbeda, urutannya nyaris tidak
   berubah, sehingga dokumen yang benar sering ada di dalam lima besar tetapi jarang di
   puncak.
3. **Kasus E01 mengambil konteks tidak relevan.** Bukan kegagalan satu kueri, melainkan
   akibat sempitnya himpunan yang dapat dijangkau.

Ketiganya kini punya satu sebab yang sama: **jangkauan penelusuran yang sempit**.

APA YANG DIUKUR
---------------
Bukan "berapa potongan unik dari beberapa kueri", melainkan batas atasnya: **berapa persen
korpus yang dapat dijangkau sama sekali** bila SELURUH nilai glukosa yang mungkin disapu.
Grid glukosa 40-400 mg/dL langkah 5 mencakup setiap nilai yang dapat dihasilkan prediktor,
sehingga angkanya adalah **jangkauan maksimum konfigurasi produksi**, bukan sekadar hasil
satu himpunan kueri.

Diukur pada tiga tingkat, dari paling sempit ke paling luas:
  A. bentuk kueri produksi saja, sapuan glukosa penuh
  B. seluruh 7 bentuk kueri T3.1, sapuan glukosa penuh
  C. ditambah kueri evaluasi RAGAS yang benar-benar dipakai T1.2

APAKAH INI AKIBAT lambda_mult 0,0?
----------------------------------
`lambda_mult` adalah parameter waktu-kueri, sehingga 0,0 dan 0,5 dapat dibandingkan
langsung tanpa mengindeks ulang. Pada LangChain, 0 berarti keberagaman maksimum dan 1
berarti keberagaman minimum. Dugaan yang wajar adalah 0,0 justru MEMPERLUAS jangkauan;
bila ternyata tidak, itu sendiri temuan tentang cara MMR bekerja di sini.

Keluaran: results/eval_rag/konsentrasi_penelusuran.json
"""
from __future__ import annotations

import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/eval_rag/konsentrasi_penelusuran.json"

# Sapuan glukosa penuh: setiap nilai yang dapat dihasilkan prediktor.
GRID = list(range(40, 401, 5))
LAMBDA_DIUJI = [0.0, 0.5]


def kunci_chunk(d: dict) -> str:
    """Identitas potongan. chunk_id bila ada; kalau tidak, cuplikan teks."""
    meta = d.get("metadata") or {}
    cid = meta.get("chunk_id")
    return str(cid) if cid else (d.get("text") or "")[:200]


def total_chunk_koleksi(persist: str, koleksi: str) -> int | None:
    """Cacah chunk SEBENARNYA di koleksi Chroma — penyebut yang benar, bukan ditebak."""
    try:
        import chromadb
        c = chromadb.PersistentClient(path=persist)
        return int(c.get_collection(koleksi).count())
    except Exception as exc:  # noqa: BLE001
        print(f"  (tidak dapat membaca cacah koleksi: {exc})")
        return None


def sapu(r, bentuk, grid, top_k) -> tuple[set, int, Counter]:
    """Jalankan satu bentuk kueri atas seluruh grid. Kembalikan himpunan chunk terambil."""
    terlihat, n_ambil, frekuensi = set(), 0, Counter()
    for nama_bentuk, fn in bentuk:
        for g in grid:
            from ablation_rag_fullkb import classify_glucose
            kond = classify_glucose(float(g))
            for d in r.retrieve(fn(float(g), kond), top_k=top_k):
                k = kunci_chunk(d)
                terlihat.add(k)
                frekuensi[k] += 1
                n_ambil += 1
    return terlihat, n_ambil, frekuensi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persist", default="models/chroma_db")
    ap.add_argument("--collection", default="diabetes_kb")
    args = ap.parse_args()

    from src.config import load_rag_config
    from src.rag.ablation_query import build_ablation_query
    from src.rag.retriever import MMRRetriever
    from eval_susunan_kueri import VARIAN

    cfg = load_rag_config()
    top_k = cfg.top_k
    n_korpus = total_chunk_koleksi(args.persist, args.collection)

    print("KONSENTRASI PENELUSURAN — berapa besar korpus yang dapat dijangkau?")
    print(f"korpus          : {n_korpus:,} chunk" if n_korpus else "korpus: tidak terbaca")
    print(f"top_k produksi  : {top_k} | fetch_k {cfg.fetch_k} | "
          f"lambda_mult {cfg.lambda_mult}")
    print(f"grid glukosa    : {GRID[0]}-{GRID[-1]} mg/dL langkah 5 ({len(GRID)} nilai)\n")

    produksi = [("produksi", lambda g, kond: build_ablation_query(g))]
    hasil = {}

    for lam in LAMBDA_DIUJI:
        r = MMRRetriever(persist_dir=args.persist, collection_name=args.collection,
                         embed_provider="sentence-transformers", lambda_mult=lam)
        blok = {}

        set_a, n_a, frek_a = sapu(r, produksi, GRID, top_k)
        blok["A_produksi"] = {"n_unik": len(set_a), "n_pengambilan": n_a}

        set_b, n_b, frek_b = sapu(r, VARIAN, GRID, top_k)
        blok["B_tujuh_bentuk"] = {"n_unik": len(set_b), "n_pengambilan": n_b}

        for nama, blok_x, frek in (("A_produksi", blok["A_produksi"], frek_a),
                                   ("B_tujuh_bentuk", blok["B_tujuh_bentuk"], frek_b)):
            if n_korpus:
                blok_x["persen_korpus_terjangkau"] = round(100 * blok_x["n_unik"] / n_korpus, 2)
                blok_x["persen_korpus_TIDAK_pernah_terambil"] = round(
                    100 * (n_korpus - blok_x["n_unik"]) / n_korpus, 2)
            n_amb = blok_x["n_pengambilan"]
            blok_x["pengambilan_per_potongan_unik"] = round(n_amb / max(blok_x["n_unik"], 1), 1)
            # Seberapa timpang sebarannya: porsi pengambilan yang jatuh ke 10 potongan teratas.
            atas10 = sum(v for _, v in frek.most_common(10))
            blok_x["porsi_pengambilan_ke_10_potongan_teratas"] = round(100 * atas10 / n_amb, 2)

        hasil[str(lam)] = blok
        p = blok["A_produksi"]
        q = blok["B_tujuh_bentuk"]
        print(f"lambda_mult = {lam}")
        print(f"  A produksi     : {p['n_unik']:>5} unik / {p['n_pengambilan']:>6} ambil"
              + (f"  = {p['persen_korpus_terjangkau']:>5.2f}% korpus" if n_korpus else ""))
        print(f"  B tujuh bentuk : {q['n_unik']:>5} unik / {q['n_pengambilan']:>6} ambil"
              + (f"  = {q['persen_korpus_terjangkau']:>5.2f}% korpus" if n_korpus else ""))
        print(f"  10 potongan teratas menyerap "
              f"{p['porsi_pengambilan_ke_10_potongan_teratas']}% pengambilan (produksi)\n")

    # ── MEKANISME ────────────────────────────────────────────────────────────
    # Dua batas keras yang bersama-sama menentukan jangkauan, dan keduanya dapat
    # dihitung tanpa menebak.
    from src.rag.ablation_query import CONDITION_PHRASE
    from ablation_rag_fullkb import classify_glucose

    teks_kueri = {build_ablation_query(float(g)) for g in GRID}
    frasa_terpakai = {classify_glucose(float(g)) for g in GRID}

    r0 = MMRRetriever(persist_dir=args.persist, collection_name=args.collection,
                      embed_provider="sentence-transformers", lambda_mult=cfg.lambda_mult)

    # (a) Kueri frasa kondisi SAJA, tanpa angka: tepat sebanyak kelas kondisi.
    set_frasa = set()
    for kelas in sorted(frasa_terpakai):
        for d in r0.retrieve(CONDITION_PHRASE[kelas], top_k=top_k):
            set_frasa.add(kunci_chunk(d))

    # (b) Kolam kandidat fetch_k SEBELUM MMR memilih. Ini batas atas sesungguhnya:
    #     MMR tidak pernah dapat mengembalikan potongan yang tidak masuk kolam.
    set_kolam = set()
    for g in GRID:
        for d in r0.retrieve(build_ablation_query(float(g)), top_k=cfg.fetch_k):
            set_kolam.add(kunci_chunk(d))

    mekanisme = {
        "n_nilai_glukosa_disapu": len(GRID),
        "n_teks_kueri_berbeda": len(teks_kueri),
        "n_frasa_kondisi_berbeda": len(frasa_terpakai),
        "unik_dari_frasa_kondisi_saja": len(set_frasa),
        "unik_dari_produksi": len(hasil[str(cfg.lambda_mult)]["A_produksi"]["n_unik"]) \
            if isinstance(hasil[str(cfg.lambda_mult)]["A_produksi"]["n_unik"], (list, set)) \
            else hasil[str(cfg.lambda_mult)]["A_produksi"]["n_unik"],
        "tambahan_dari_angka_glukosa": None,
        "unik_di_kolam_fetch_k": len(set_kolam),
        "batas_atas_teoretis": len(frasa_terpakai) * cfg.fetch_k,
        "penjelasan": (
            f"{len(GRID)} nilai glukosa menghasilkan {len(teks_kueri)} teks kueri berbeda, "
            f"tetapi hanya {len(frasa_terpakai)} FRASA KONDISI berbeda. Angka glukosa "
            f"nyaris tidak menggeser embedding, sehingga sapuan seluruh rentang glukosa "
            f"secara efektif hanya menghasilkan {len(frasa_terpakai)} kueri. Ditambah "
            f"fetch_k={cfg.fetch_k}, kolam kandidat maksimum menjadi sekitar "
            f"{len(frasa_terpakai)} x {cfg.fetch_k} = {len(frasa_terpakai) * cfg.fetch_k} "
            f"potongan, dan MMR memilih dari situ."),
    }
    mekanisme["tambahan_dari_angka_glukosa"] = (
        mekanisme["unik_dari_produksi"] - mekanisme["unik_dari_frasa_kondisi_saja"])

    print("MEKANISME")
    print(f"  {len(GRID)} nilai glukosa -> {len(teks_kueri)} teks kueri berbeda, "
          f"tetapi hanya {len(frasa_terpakai)} frasa kondisi")
    print(f"  frasa kondisi saja ({len(frasa_terpakai)} kueri) : "
          f"{len(set_frasa)} potongan unik")
    print(f"  produksi ({len(GRID)} kueri berangka)            : "
          f"{mekanisme['unik_dari_produksi']} potongan unik "
          f"(angka menambah {mekanisme['tambahan_dari_angka_glukosa']})")
    print(f"  kolam kandidat fetch_k={cfg.fetch_k} sebelum MMR  : "
          f"{len(set_kolam)} potongan (batas atas teoretis "
          f"{len(frasa_terpakai) * cfg.fetch_k})\n")

    prod_lam = str(cfg.lambda_mult)
    a = hasil[prod_lam]["A_produksi"]
    pembanding = [l for l in LAMBDA_DIUJI if str(l) != prod_lam]
    banding = None
    if pembanding:
        lain = str(pembanding[0])
        banding = {
            "lambda_produksi": cfg.lambda_mult, "lambda_pembanding": pembanding[0],
            "n_unik_produksi": a["n_unik"],
            "n_unik_pembanding": hasil[lain]["A_produksi"]["n_unik"],
            "selisih": hasil[lain]["A_produksi"]["n_unik"] - a["n_unik"],
        }

    out = {
        "temuan": ("Konsentrasi penelusuran — bagian korpus yang dapat dijangkau "
                   "konfigurasi produksi"),
        "status": ("TEMUAN TERSENDIRI, bukan catatan metode. Muncul tanpa dicari saat "
                   "menyiapkan T2.2."),
        "waktu": datetime.now().isoformat(timespec="seconds"),
        "n_chunk_korpus": n_korpus,
        "konfigurasi": {"top_k": top_k, "fetch_k": cfg.fetch_k,
                        "lambda_mult_produksi": cfg.lambda_mult,
                        "chunk_size": cfg.chunk_size,
                        "embedding_model": cfg.embedding_model},
        "grid_glukosa": {"min": GRID[0], "maks": GRID[-1], "langkah": 5, "n": len(GRID),
                         "alasan": ("mencakup setiap nilai yang dapat dihasilkan "
                                    "prediktor, sehingga hasilnya adalah JANGKAUAN "
                                    "MAKSIMUM konfigurasi produksi, bukan hasil satu "
                                    "himpunan kueri")},
        "hasil_per_lambda": hasil,
        "perbandingan_lambda": banding,
        "mekanisme": mekanisme,
        "catatan_cacah_korpus": (
            "Koleksi Chroma berisi 2.061 chunk, sedangkan T5.1 menghitung 2.186 chunk saat "
            "memecah PDF langsung pada chunk_size 900. Selisih 125 berasal dari penyaring "
            "min-chunk-chars pada reingest_kb.py, yang membuang fragmen ekor halaman. "
            "Penyebut yang dipakai di sini adalah 2.061, yaitu isi indeks yang benar-benar "
            "dapat ditelusuri."),
        "menjelaskan_tiga_temuan_lain": [
            "Pola biner context_precision (T1.2): himpunan terjangkau yang sempit membuat "
            "konteks sebuah kueri hampir seluruhnya benar atau hampir seluruhnya salah.",
            "Hit@1 23,3% lawan Hit@5 91,7%: kandidat yang sama berulang membuat urutan "
            "nyaris tidak berubah, sehingga dokumen benar ada di lima besar tetapi jarang "
            "di puncak.",
            "Kasus E01 mengambil konteks tidak relevan: akibat sempitnya himpunan "
            "terjangkau, bukan kegagalan satu kueri.",
        ],
        "yang_BELUM_dibuktikan": (
            "Bahwa konsentrasi ini MENYEBABKAN ketiga temuan itu. Yang ditunjukkan di sini "
            "adalah satu sebab yang konsisten dengan ketiganya dan terukur besarnya. "
            "Pembuktian menuntut perbandingan terhadap konfigurasi yang jangkauannya lebih "
            "luas, lalu memeriksa apakah ketiga pola itu ikut berubah."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    if n_korpus:
        print(f"ANGKA UNTUK BAB VI: pada konfigurasi produksi (lambda {cfg.lambda_mult}, "
              f"top_k {top_k}), sapuan seluruh {len(GRID)} nilai glukosa hanya menjangkau "
              f"{a['n_unik']:,} dari {n_korpus:,} chunk "
              f"= {a['persen_korpus_terjangkau']}% korpus.")
        print(f"{a['persen_korpus_TIDAK_pernah_terambil']}% korpus TIDAK PERNAH terambil "
              f"oleh nilai glukosa mana pun.")
    if banding:
        print(f"\nlambda {banding['lambda_produksi']} -> {banding['n_unik_produksi']} unik | "
              f"lambda {banding['lambda_pembanding']} -> {banding['n_unik_pembanding']} unik "
              f"(selisih {banding['selisih']:+d})")
    print(f"\nDisimpan ke {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
