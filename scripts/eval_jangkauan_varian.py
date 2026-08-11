"""T3.3 — jangkauan varian kueri, dan bentuk kueri yang BENAR-BENAR dipakai aplikasi.

Prapendaftaran: docs/PRAPENDAFTARAN_T3.3.md, ditulis sebelum skrip ini ada.

DUA PERTANYAAN, SATU SKRIP
--------------------------
Keduanya menyentuh berkas yang sama (`VARIAN` T3.1 dan indeks Chroma produksi), sehingga
memisahkannya menjadi dua skrip hanya akan menggandakan pemuatan model embedding.

**Pertanyaan 1 — pertukaran peringkat lawan jangkauan.** T3.1 menunjukkan `hanya_kondisi`
menaikkan Hit@1 dari 42,2% ke 71,1%. TEMUAN jangkauan menunjukkan frasa kondisi sendirian
hanya menjangkau 14 potongan sedangkan bentuk produksi menjangkau 24. Kedua besaran itu
belum pernah diukur bersama, dan keputusan mengadopsi `hanya_kondisi` ke produksi
bergantung pada keduanya sekaligus.

**Pertanyaan 2 — bentuk kueri aplikasi belum pernah diukur.** Yang diukur T3.1 adalah
`build_ablation_query()`. Kueri retrieval aplikasi dibangun
`PredictionConditionedQueryBuilder._primary_query()` dan berbeda jauh: numeral empat kali,
klausa tren, faktor kontribusi, dan ditutup kalimat tanya. Seluruh angka penelusuran di
laporan karena itu mengukur bentuk yang tidak pernah dijalankan aplikasi.

SET PELAPORAN T3.1 TIDAK DISENTUH
---------------------------------
Set pelaporan (`ohio_591`, `ohio_596`) sudah dipakai sekali oleh T3.1. Menjalankan lengan
kedelapan di atasnya melanggar protokol Bagian C butir 5 dan membatalkan keabsahan T3.1
itu sendiri. Bagian B skrip ini karena itu berjalan pada **set penyetelan saja**, dan
hasilnya adalah pengukuran susulan — bukan baris kedelapan tabel T3.1.

KETERBATASAN K1 BERLAKU PENUH
-----------------------------
Relevansi berasal dari `classify_chunk()`, pelabel kata kunci. Bagian C menghitung bobot
kata kunci tiap bentuk justru supaya terlihat mana peringkat yang dapat dibaca dan mana
yang hanya memantulkan kosakata pelabelnya sendiri.

Keluaran: results/eval_rag/jangkauan_varian.json
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch sebelum numpy/pandas — WinError 1114)
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Dipakai ulang, tidak ditulis ulang. Menyalin GRID atau kunci_chunk() ke sini akan
# membuat dua sumber kebenaran, dan angka jangkauan T3.3 tidak lagi sebanding dengan
# angka TEMUAN yang dihasilkan eval_konsentrasi_penelusuran.py.
from eval_konsentrasi_penelusuran import GRID, kunci_chunk, total_chunk_koleksi  # noqa: E402
from eval_susunan_kueri import VARIAN, bangun_kasus, nilai_varian  # noqa: E402
from ablation_rag_fullkb import KW, classify_chunk, classify_glucose, ndcg_at_k  # noqa: E402
from tuning_protocol import KRITERIA, catat  # noqa: E402

OUT = ROOT / "results/eval_rag/jangkauan_varian.json"
LOG_PENYETELAN = ROOT / "results/tuning_log.json"

TOP_K = 5
SEED = 42
# Horizon 30 menit (default_horizon 6 x sampling 5 mnt), sepadan dengan HORIZON=6 pada
# T3.1 — BUKAN 60 menit yang menjadi bawaan build_conditioned_query().
HORIZON_MIN = 30
GLUKOSA_CONTOH = {"hipoglikemia": 58.0, "normal": 120.0, "hiperglikemia": 230.0}


# ── Lengan `aplikasi`: bentuk kueri yang benar-benar dijalankan aplikasi ─────────────
def _kueri_aplikasi(prediksi: float, sekarang: float) -> str:
    """Kueri retrieval aplikasi sungguhan, lewat jalur produksi yang tidak diubah.

    Sengaja memanggil ``build_conditioned_query`` alih-alih menyusun ulang stringnya di
    sini: kalau _primary_query() berubah suatu hari, angka T3.3 ikut berubah dan tidak
    diam-diam mengukur bentuk yang sudah tidak ada.
    """
    from src.rag.conditioned_query import build_conditioned_query
    return build_conditioned_query(
        patient_id="T3.3",
        current_glucose=float(sekarang),
        predicted_glucose=float(prediksi),
        prediction_horizon_minutes=HORIZON_MIN,
    ).primary_query


def v_aplikasi_delta0(g, c):
    """Sebanding dengan tujuh varian lain: hanya angka prediksi yang berubah."""
    return _kueri_aplikasi(g, g)


# ── Bagian A: jangkauan ──────────────────────────────────────────────────────────────
def sapu_satu_bentuk(r, fn, grid, top_k) -> tuple[set, int, Counter]:
    """Satu bentuk kueri atas seluruh grid glukosa. Sengaja PER BENTUK.

    ``sapu()`` pada eval_konsentrasi_penelusuran.py menggabungkan seluruh bentuk menjadi
    satu himpunan, yang menjawab "berapa jangkauan gabungan" — bukan pertanyaan di sini.
    """
    terlihat, n_ambil, frekuensi = set(), 0, Counter()
    for g in grid:
        kond = classify_glucose(float(g))
        for d in r.retrieve(fn(float(g), kond), top_k=top_k):
            k = kunci_chunk(d)
            terlihat.add(k)
            frekuensi[k] += 1
            n_ambil += 1
    return terlihat, n_ambil, frekuensi


def ringkas_jangkauan(terlihat: set, n_ambil: int, frekuensi: Counter,
                      n_korpus: int | None) -> dict:
    atas10 = sum(v for _, v in frekuensi.most_common(10))
    blok = {
        "n_unik": len(terlihat),
        "n_pengambilan": n_ambil,
        "pengambilan_per_potongan_unik": round(n_ambil / max(len(terlihat), 1), 1),
        "porsi_pengambilan_ke_10_potongan_teratas": round(100 * atas10 / max(n_ambil, 1), 2),
    }
    if n_korpus:
        blok["persen_korpus_terjangkau"] = round(100 * len(terlihat) / n_korpus, 2)
        blok["persen_korpus_TIDAK_pernah_terambil"] = round(
            100 * (n_korpus - len(terlihat)) / n_korpus, 2)
    return blok


# ── Bagian C: bobot kata kunci pelabel ───────────────────────────────────────────────
def bobot_kata_kunci(fn) -> dict:
    """Tumpang tindih kosakata antara kueri sebuah bentuk dan pelabel classify_chunk().

    Cara hitungnya IDENTIK dengan results/eval_rag/tumpang_tindih_kata_kunci.json: untuk
    tiap kelas, kueri kelas itu dicocokkan dengan daftar kata kunci kelas itu sendiri.
    Spearman(bobot, MRR) = 0,964 pada T3.1, sehingga peringkat antar-bentuk yang bobotnya
    BERBEDA tidak dapat dibaca sebagai selisih bentuk kueri.
    """
    per_kelas, total = {}, 0
    for kelas, g in GLUKOSA_CONTOH.items():
        teks = fn(g, kelas).lower()
        cocok = [(kw, w) for kw, w in KW[kelas] if kw in teks]
        bobot = sum(w for _, w in cocok)
        per_kelas[kelas] = {"n_kata_kunci": len(cocok), "bobot": bobot,
                            "kata_kunci_cocok": [kw for kw, _ in cocok]}
        total += bobot
    return {"per_kelas": per_kelas, "total_bobot": total}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persist", default="models/chroma_db")
    ap.add_argument("--collection", default="diabetes_kb")
    ap.add_argument("--lewati-peringkat", action="store_true",
                    help="hanya Bagian A dan C (tanpa melatih RF)")
    args = ap.parse_args()

    from src.config import load_rag_config
    from src.rag.retriever import MMRRetriever

    cfg_rag = load_rag_config()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    n_korpus = total_chunk_koleksi(args.persist, args.collection)

    # Delapan bentuk: tujuh varian T3.1 + bentuk aplikasi.
    bentuk = list(VARIAN) + [("aplikasi_delta0", v_aplikasi_delta0)]

    print("T3.3 — jangkauan varian kueri dan bentuk kueri aplikasi")
    print(f"korpus         : {n_korpus:,} chunk" if n_korpus else "korpus: tidak terbaca")
    print(f"konfigurasi    : top_k {cfg_rag.top_k} | fetch_k {cfg_rag.fetch_k} | "
          f"lambda_mult {cfg_rag.lambda_mult} | chunk_size {cfg_rag.chunk_size}")
    print(f"grid glukosa   : {GRID[0]}-{GRID[-1]} mg/dL langkah 5 ({len(GRID)} nilai)")
    print(f"bentuk diuji   : {len(bentuk)} (tujuh varian T3.1 + bentuk aplikasi)")
    print("Prapendaftaran : docs/PRAPENDAFTARAN_T3.3.md\n")

    # ── BAGIAN C lebih dulu: tanpa retrieval, dan menentukan cara Bagian B dibaca ────
    print("BAGIAN C — bobot kata kunci pelabel (tumpang tindih dengan classify_chunk)")
    bobot = {nama: bobot_kata_kunci(fn) for nama, fn in bentuk}
    for nama, b in bobot.items():
        print(f"  {nama:<18} total {b['total_bobot']:>3}  "
              + " ".join(f"{k[:5]}={v['bobot']}" for k, v in b["per_kelas"].items()))
    b_apl = bobot["aplikasi_delta0"]["total_bobot"]
    b_tanya = bobot["pertanyaan"]["total_bobot"]
    kebal = (b_apl == b_tanya)
    print(f"\n  Prapendaftaran D3 memperkirakan bobot bentuk aplikasi = 6 "
          f"-> TERUKUR {b_apl} : {'TERPENUHI' if b_apl == 6 else 'TIDAK TERPENUHI'}")
    print(f"  aplikasi ({b_apl}) vs pertanyaan ({b_tanya}) -> perbandingan "
          f"{'KEBAL kontaminasi' if kebal else 'TIDAK kebal — peringkat tidak boleh dinarasikan'}\n")

    # ── BAGIAN A: jangkauan ─────────────────────────────────────────────────────────
    r = MMRRetriever(persist_dir=args.persist, collection_name=args.collection,
                     embed_provider="sentence-transformers")

    print("BAGIAN A — jangkauan tiap bentuk atas sapuan glukosa penuh")
    print(f"  {'bentuk':<18}{'unik':>7}{'% korpus':>11}{'ambil':>8}"
          f"{'per unik':>10}{'% ke 10 teratas':>17}{'kueri unik':>12}")
    jangkauan = {}
    t0 = time.time()
    for nama, fn in bentuk:
        terlihat, n_ambil, frek = sapu_satu_bentuk(r, fn, GRID, cfg_rag.top_k)
        blok = ringkas_jangkauan(terlihat, n_ambil, frek, n_korpus)
        # Berapa TEKS kueri berbeda yang sebenarnya dihasilkan bentuk ini atas 73 nilai
        # glukosa. Inilah tuas yang sesungguhnya: hanya_kondisi runtuh menjadi 3.
        blok["n_teks_kueri_berbeda"] = len(
            {fn(float(g), classify_glucose(float(g))) for g in GRID})
        jangkauan[nama] = blok
        print(f"  {nama:<18}{blok['n_unik']:>7}"
              f"{blok.get('persen_korpus_terjangkau', float('nan')):>10.2f}%"
              f"{blok['n_pengambilan']:>8}{blok['pengambilan_per_potongan_unik']:>10.1f}"
              f"{blok['porsi_pengambilan_ke_10_potongan_teratas']:>16.2f}%"
              f"{blok['n_teks_kueri_berbeda']:>12}", flush=True)
    print(f"  [{time.time() - t0:.0f} dtk]\n")

    prod = jangkauan["produksi"]
    hk = jangkauan["hanya_kondisi"]
    pertukaran = {
        "produksi": {"n_unik": prod["n_unik"],
                     "persen_korpus": prod.get("persen_korpus_terjangkau"),
                     "porsi_10_teratas": prod["porsi_pengambilan_ke_10_potongan_teratas"],
                     "mrr_T3.1_set_penyetelan": 0.6211, "hit@1_T3.1_set_penyetelan": 0.4222},
        "hanya_kondisi": {"n_unik": hk["n_unik"],
                          "persen_korpus": hk.get("persen_korpus_terjangkau"),
                          "porsi_10_teratas": hk["porsi_pengambilan_ke_10_potongan_teratas"],
                          "mrr_T3.1_set_penyetelan": 0.7111,
                          "hit@1_T3.1_set_penyetelan": 0.7111},
        "selisih_n_unik": hk["n_unik"] - prod["n_unik"],
        "selisih_persen_relatif": round(
            100 * (hk["n_unik"] - prod["n_unik"]) / max(prod["n_unik"], 1), 1),
        "prapendaftaran_D1_perkiraan_n_unik": 14,
        "prapendaftaran_D1_terpenuhi": bool(hk["n_unik"] == 14),
    }
    print(f"PERTUKARAN peringkat lawan jangkauan (bahan keputusan menggantung #6):")
    print(f"  produksi      : {prod['n_unik']:>3} potongan unik | "
          f"MRR 0,6211 · Hit@1 0,4222 (T3.1, set penyetelan)")
    print(f"  hanya_kondisi : {hk['n_unik']:>3} potongan unik | "
          f"MRR 0,7111 · Hit@1 0,7111 (T3.1, set penyetelan)")
    print(f"  Prapendaftaran D1 memperkirakan 14 -> TERUKUR {hk['n_unik']} : "
          f"{'TERPENUHI' if hk['n_unik'] == 14 else 'TIDAK TERPENUHI'}\n")

    # ── BAGIAN B: peringkat lengan aplikasi, SET PENYETELAN SAJA ────────────────────
    peringkat = None
    if not args.lewati_peringkat:
        df_pid = pd.read_csv(ROOT / "data/raw/ohio_t1dm_merged.csv", usecols=["patient_id"])
        pids = sorted(df_pid["patient_id"].unique().tolist())
        latih, setel, lapor = pids[:-4], pids[-4:-2], pids[-2:]

        print("BAGIAN B — peringkat bentuk aplikasi, SET PENYETELAN SAJA")
        print(f"  latih={len(latih)} | PENYETELAN={setel}")
        print(f"  set PELAPORAN {lapor} sengaja TIDAK disentuh — sudah dipakai sekali "
              f"oleh T3.1 (Bagian C butir 5)")
        print(f"  kriteria diimpor dari tuning_protocol: {KRITERIA.upper()}")

        t0 = time.time()
        kasus = bangun_kasus(cfg, latih, setel, np.random.default_rng(SEED))
        print(f"  {len(kasus)} kasus {dict(kasus['cond_actual'].value_counts())} "
              f"[{time.time() - t0:.0f} dtk]\n")

        # Lengan kedua: current_glucose SEBENARNYA dari kasus, setia pada kueri aplikasi.
        def nilai_aplikasi_penuh(r, kasus) -> dict:
            hit1, mrr, ndcg = [], [], []
            for _, c in kasus.iterrows():
                q = _kueri_aplikasi(float(c["predicted"]), float(c["current"]))
                docs = r.retrieve(q, top_k=TOP_K)
                topik = [classify_chunk(d["text"]) for d in docs]
                exp = str(c["cond_actual"])
                rank = (topik.index(exp) + 1) if exp in topik else 0
                hit1.append(int(bool(topik) and topik[0] == exp))
                mrr.append((1.0 / rank) if rank else 0.0)
                ndcg.append(ndcg_at_k([1 if t == exp else 0 for t in topik], TOP_K))
            return {"n": int(len(kasus)),
                    "hit@1": round(float(np.mean(hit1)), 4),
                    "mrr": round(float(np.mean(mrr)), 4),
                    "ndcg@5": round(float(np.mean(ndcg)), 4),
                    "_mrr_per_kasus": [float(x) for x in mrr]}

        # Acuan dijalankan ULANG di sini, tidak dikutip dari susunan_kueri.json: uji
        # berpasangan menuntut MRR per kasus, dan sekaligus memeriksa bahwa kasusnya
        # benar-benar sama (angka harus cocok sampai empat desimal dengan T3.1).
        print(f"  {'lengan':<20}{'MRR':>9}{'Hit@1':>9}{'nDCG@5':>9}")
        hasil = {}
        for nama, fn in [("produksi", dict(VARIAN)["produksi"]),
                         ("pertanyaan", dict(VARIAN)["pertanyaan"]),
                         ("aplikasi_delta0", v_aplikasi_delta0)]:
            s = nilai_varian(r, kasus, fn)
            hasil[nama] = s
            print(f"  {nama:<20}{s['mrr']:>9.4f}{s['hit@1']:>9.4f}{s['ndcg@5']:>9.4f}",
                  flush=True)
        s = nilai_aplikasi_penuh(r, kasus)
        hasil["aplikasi_penuh"] = s
        print(f"  {'aplikasi_penuh':<20}{s['mrr']:>9.4f}{s['hit@1']:>9.4f}"
              f"{s['ndcg@5']:>9.4f}", flush=True)

        uji = {}
        for a, b in (("aplikasi_delta0", "pertanyaan"), ("aplikasi_penuh", "pertanyaan"),
                     ("aplikasi_delta0", "produksi"), ("aplikasi_penuh", "produksi"),
                     ("aplikasi_penuh", "aplikasi_delta0")):
            try:
                p = float(stats.wilcoxon(hasil[a]["_mrr_per_kasus"],
                                         hasil[b]["_mrr_per_kasus"]).pvalue)
            except ValueError:
                p = float("nan")
            uji[f"{a}_vs_{b}"] = {
                "selisih_mrr": round(hasil[a]["mrr"] - hasil[b]["mrr"], 4),
                "wilcoxon_p": None if p != p else round(p, 6),
                "signifikan": bool(p == p and p < 0.05),
                "kebal_kontaminasi": bool(
                    bobot.get(a, bobot["aplikasi_delta0"])["total_bobot"]
                    == bobot.get(b, bobot["aplikasi_delta0"])["total_bobot"]),
            }
        print("\n  Uji berpasangan (MRR per kasus, Wilcoxon):")
        for k, v in uji.items():
            print(f"    {k:<38} {v['selisih_mrr']:>+8.4f} | p={v['wilcoxon_p']} | "
                  f"{'KEBAL' if v['kebal_kontaminasi'] else 'bobot berbeda'}")

        # Verifikasi acuan: MRR produksi di sini HARUS sama dengan T3.1 set penyetelan.
        acuan_t31 = 0.6211
        cocok = abs(hasil["produksi"]["mrr"] - acuan_t31) < 5e-4
        print(f"\n  Verifikasi kasus identik dengan T3.1: MRR produksi "
              f"{hasil['produksi']['mrr']:.4f} vs {acuan_t31} -> "
              f"{'COCOK' if cocok else 'TIDAK COCOK — kasus berbeda, angka tidak sepadan'}")

        mrr_apl = min(hasil["aplikasi_delta0"]["mrr"], hasil["aplikasi_penuh"]["mrr"])
        mrr_apl_maks = max(hasil["aplikasi_delta0"]["mrr"], hasil["aplikasi_penuh"]["mrr"])
        d2_penuh = bool(0.2276 < mrr_apl and mrr_apl_maks < 0.4744)
        d2_sebagian = bool(not d2_penuh and mrr_apl < 0.4744)
        print(f"  Prapendaftaran D2 (MRR di rentang 0,2276-0,4744, kedua lengan) -> "
              f"{'TERPENUHI' if d2_penuh else ('TERPENUHI SEBAGIAN' if d2_sebagian else 'TIDAK TERPENUHI')}")

        for v in hasil.values():
            v.pop("_mrr_per_kasus", None)
        peringkat = {
            "catatan_protokol": (
                "SET PENYETELAN SAJA. Set pelaporan T3.1 (ohio_591, ohio_596) tidak "
                "disentuh karena sudah dipakai sekali. Angka di sini adalah pengukuran "
                "susulan, BUKAN baris kedelapan tabel T3.1, dan tidak boleh disandingkan "
                "dengan kolom set pelaporan T3.1."),
            "set_latih": latih, "set_penyetelan": setel,
            "set_pelaporan_TIDAK_disentuh": lapor,
            "kriteria_ditetapkan_di_muka": KRITERIA,
            "n_kasus": int(len(kasus)),
            "verifikasi_kasus_identik_dengan_T3.1": {
                "mrr_produksi_terukur": hasil["produksi"]["mrr"],
                "mrr_produksi_T3.1": acuan_t31, "cocok": cocok,
                "arti_bila_tidak_cocok": ("kasus, benih, atau pembagian pasien berbeda; "
                                          "seluruh angka Bagian B tidak sepadan dengan "
                                          "T3.1 dan tidak boleh disandingkan"),
            },
            "hasil": hasil, "uji_berpasangan": uji,
            "prapendaftaran_D2_rentang": [0.2276, 0.4744],
            "prapendaftaran_D2_terpenuhi": d2_penuh,
            "prapendaftaran_D2_terpenuhi_sebagian": d2_sebagian,
        }

    contoh = {nama: fn(58.0, "hipoglikemia") for nama, fn in bentuk}
    contoh["aplikasi_penuh"] = _kueri_aplikasi(58.0, 112.0)

    out = {
        "percobaan": "T3.3 — jangkauan varian kueri dan bentuk kueri aplikasi",
        "waktu": datetime.now().isoformat(timespec="seconds"),
        "prapendaftaran": "docs/PRAPENDAFTARAN_T3.3.md",
        "n_chunk_korpus": n_korpus,
        "konfigurasi": {"top_k": cfg_rag.top_k, "fetch_k": cfg_rag.fetch_k,
                        "lambda_mult": cfg_rag.lambda_mult,
                        "chunk_size": cfg_rag.chunk_size,
                        "embedding_model": cfg_rag.embedding_model,
                        "horizon_menit": HORIZON_MIN},
        "grid_glukosa": {"min": GRID[0], "maks": GRID[-1], "langkah": 5, "n": len(GRID)},
        "contoh_kueri": contoh,
        "A_jangkauan": jangkauan,
        "A_pertukaran_peringkat_lawan_jangkauan": pertukaran,
        "B_peringkat_bentuk_aplikasi": peringkat,
        "C_bobot_kata_kunci_pelabel": bobot,
        "C_kebal_kontaminasi": {
            "aplikasi_vs_pertanyaan": kebal,
            "bobot_aplikasi": b_apl, "bobot_pertanyaan": b_tanya,
            "prapendaftaran_D3_perkiraan": 6,
            "prapendaftaran_D3_terpenuhi": bool(b_apl == 6),
            "aturan": ("Spearman(bobot, MRR) = 0,964 pada T3.1. Peringkat antar-bentuk "
                       "yang bobotnya BERBEDA tidak boleh dinarasikan sebagai selisih "
                       "bentuk kueri; hanya pasangan berbobot sama yang dapat dibaca."),
        },
        "keterbatasan_K1": ("Relevansi dari classify_chunk(), pelabel kata kunci, bukan "
                            "penilaian manusia. Selama T2.2 belum diisi, tidak satu pun "
                            "angka di sini boleh dinyatakan mantap."),
        "keterbatasan_lengan_aplikasi": (
            "Pada Bagian A, lengan aplikasi dibangun dengan current_glucose = "
            "predicted_glucose (delta 0, tren stabil), tanpa IOB/COB aktif dan tanpa "
            "interval konformal, supaya hanya angka prediksi yang berubah dan ia "
            "sebanding dengan tujuh bentuk lain. Kueri aplikasi sungguhan lebih beragam, "
            "sehingga angka jangkauannya adalah BATAS BAWAH. Bagian B menjalankan kedua "
            "lengan: delta0 (sebanding) dan penuh (setia pada aplikasi)."),
        "yang_TIDAK_diukur": (
            "Mutu jawaban. Bila bentuk aplikasi berperingkat buruk, yang terbukti adalah "
            "bentuk kuerinya merugikan penelusuran menurut pelabel kata kunci — bukan "
            "bahwa jawaban yang dihasilkan aplikasi lebih buruk. Itu diukur RAGAS (T1.2) "
            "dan tidak diuji di sini."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    if peringkat:
        for nama in ("aplikasi_delta0", "aplikasi_penuh"):
            catat(parameter="bentuk_kueri_aplikasi", nilai=nama,
                  metrik_penyetelan={k: peringkat["hasil"][nama][k]
                                     for k in ("n", "hit@1", "mrr", "ndcg@5")},
                  catatan=(f"T3.3 — pengukuran susulan pada set penyetelan {setel}; set "
                           f"pelaporan T3.1 tidak disentuh. Bobot kata kunci pelabel "
                           f"{b_apl} (pertanyaan {b_tanya})."),
                  log_path=LOG_PENYETELAN)

    print(f"\nDisimpan ke {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
