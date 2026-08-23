"""T8 — apakah PEMOTONGAN SENYAP itu sendiri merugikan retrieval?

PRAPENDAFTARAN, ditulis SEBELUM hasil dilihat (16 Agustus 2026).

MENGAPA PERCOBAAN INI SAH SEDANGKAN T7 TIDAK BISA MEMUTUSKAN.
T7 membandingkan varian yang batas potongannya BERBEDA. Karena label kebenaran
dihasilkan classify_chunk() dari teks potongan itu sendiri, labelnya ikut bergeser
bersama perlakuan, dan perbandingan MRR antarvarian menjadi terkonfound
(results/eval_rag/diagnosis_metrik_chunking.json).

Di sini masalah itu TIDAK ADA. W256 dan W512 memakai chunk_size, chunk_overlap,
dan daftar pemisah yang PERSIS SAMA, sehingga potongannya identik teks demi teks
dan labelnya identik pula. Satu-satunya yang berubah adalah jendela token model
embedding. Karena itu selisih MRR-nya dapat dikaitkan pada pemotongan — inilah
uji yang diminta kesimpulan T5.1 (scripts/eval_distribusi_token.py:265) dan belum
pernah dijalankan.

DASAR ANGKA 256. Berasal dari sentence_bert_config.json milik all-MiniLM-L6-v2
("max_seq_length": 256) — PILIHAN penulis model, bukan batas arsitektur. BERT di
bawahnya ber-max_position_embeddings 512, sehingga posisi 257..512 bobotnya ADA.
Potongan terpanjang korpus produksi 443 token, jadi 512 menghapus pemotongan
seluruhnya tanpa menyentuh satu pun batas potongan.

VARIAN
  W256  chunk 900, pemisah lama, jendela 256  -> kontrol (= V0 pada T7)
  W512  chunk 900, pemisah lama, jendela 512  -> SATU faktor berubah
  W512K chunk 900, pemisah kalimat, jendela 512 -> kandidat produksi

DUGAAN
  H1 W512 menghasilkan potongan terpotong NOL (443 < 512). Konstruktif, bukan
     statistik; bila gagal berarti tombolnya tidak benar-benar terpasang.
  H2 Vektor W512 BERBEDA dari W256 untuk potongan >256 token. Bila sama, setelan
     jendela diabaikan diam-diam dan seluruh hasil batal.
  H3 Potongan <=256 token WAJIB bervektor identik di kedua jendela. Ini kontrol
     negatifnya: jendela hanya boleh berpengaruh pada yang sebelumnya terpotong.
  H4 Arah MRR TIDAK didugakan. Yang didaftarkan kriterianya: bila W512 > W256
     melebihi 0,02 maka pemotongan terbukti merugikan; bila |selisih| < 0,02 maka
     pemotongan tidak terbukti berpengaruh pada metrik ini meskipun 8,23% korpus
     tidak tervektor — dan itu temuan yang layak dilaporkan apa adanya.

Keluaran: results/eval_rag/jendela_token.json
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ablation_rag_fullkb import TOP_K, classify_chunk  # noqa: E402
from src.rag.ablation_query import build_ablation_query  # noqa: E402
from tuning_protocol import KRITERIA, bagi_dua, catat  # noqa: E402

CORPUS_TAG = os.environ.get("CORPUS_TAG", "kb12_sym")
OUT = ROOT / "results/eval_rag/jendela_token.json"
TMP = ROOT / "models/_jendela_token"

# (kode, chunk_size, overlap, pemisah_kalimat, jendela, keterangan)
VARIAN = [
    ("W256", 900, 120, False, 256, "kontrol — jendela bawaan model"),
    ("W512", 900, 120, False, 512, "SATU faktor: jendela dinaikkan, potongan identik"),
    ("W512K", 900, 120, True, 512, "kandidat produksi: jendela 512 + pemisah kalimat"),
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


def indeks_ulang(ukuran, overlap, pemisah, jendela, persist: Path) -> int:
    if persist.exists():
        shutil.rmtree(persist, ignore_errors=True)
    perintah = [
        sys.executable, str(ROOT / "scripts/reingest_kb.py"),
        "--persist", str(persist), "--collection", "diabetes_kb",
        "--chunk-size", str(ukuran), "--chunk-overlap", str(overlap),
        "--max-seq-length", str(jendela),
    ]
    if pemisah:
        perintah.append("--pemisah-kalimat")
    proc = subprocess.run(perintah, cwd=str(ROOT),
                          env=dict(os.environ, PYTHONPATH=str(ROOT)),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"reingest gagal ({jendela}):\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    import chromadb
    n = chromadb.PersistentClient(path=str(persist)).get_collection("diabetes_kb").count()
    if n == 0:
        raise RuntimeError("Indeks kosong — hasil tidak sah.")
    return n


def evaluasi(pasangan, persist: Path, jendela: int) -> dict:
    """Evaluasi dalam SUBPROSES.

    RagConfig di-cache lru_cache per proses. Menilai dua jendela berbeda dalam
    satu proses akan membuat yang kedua diam-diam memakai jendela yang pertama —
    kegagalan tanpa error yang persis sejenis dengan cacat yang sedang diteliti.
    """
    kode = (
        "import sys,os,json,warnings;warnings.filterwarnings('ignore');"
        "sys.path.insert(0,r'%s');sys.path.insert(0,r'%s');"
        "from ablation_rag_fullkb import classify_chunk,TOP_K;"
        "from src.rag.retriever import MMRRetriever;"
        "pas=json.load(open(r'%s',encoding='utf-8'));"
        "r=MMRRetriever(persist_dir=r'%s',collection_name='diabetes_kb',embed_provider='sentence-transformers');"
        "ranks=[];\n"
        "for q,e in pas:\n"
        "    t=[classify_chunk(d['text']) for d in r.retrieve(q,top_k=TOP_K)];\n"
        "    ranks.append((t.index(e)+1) if e in t else 0)\n"
        "print('JSON'+json.dumps(ranks))"
    ) % (ROOT, ROOT / "scripts", TMP / "_pasangan.json", persist)
    (TMP / "_pasangan.json").write_text(json.dumps(pasangan), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(ROOT), EMBED_MAX_SEQ_LENGTH=str(jendela))
    proc = subprocess.run([sys.executable, "-c", kode], cwd=str(ROOT), env=env,
                          capture_output=True, text=True)
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


def periksa_vektor(p256: Path, p512: Path) -> dict:
    """H2 dan H3: jendela hanya boleh mengubah vektor potongan yang tadinya terpotong."""
    import chromadb
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def ambil(p):
        c = chromadb.PersistentClient(path=str(p)).get_collection("diabetes_kb")
        r = c.get(include=["documents", "embeddings"])
        return dict(zip(r["documents"], np.array(r["embeddings"])))

    a, b = ambil(p256), ambil(p512)
    sama = [t for t in a if t in b]
    panjang = {t: len(tok.encode(t)) for t in sama}
    lebih = [t for t in sama if panjang[t] > 256]
    kurang = [t for t in sama if panjang[t] <= 256]
    cos = lambda t: float(a[t] @ b[t])
    return {
        "n_potongan_dibandingkan": len(sama),
        "n_lebih_dari_256_token": len(lebih),
        "n_maks_256_token": len(kurang),
        "H2_kemiripan_rerata_potongan_TERPOTONG": (
            round(float(np.mean([cos(t) for t in lebih])), 6) if lebih else None),
        "H2_kemiripan_maks_potongan_TERPOTONG": (
            round(float(np.max([cos(t) for t in lebih])), 6) if lebih else None),
        "H3_kemiripan_min_potongan_UTUH": (
            round(float(np.min([cos(t) for t in kurang])), 6) if kurang else None),
    }


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    skenario = muat_skenario()
    if not skenario:
        raise SystemExit(f"Skenario tidak ditemukan untuk CORPUS_TAG={CORPUS_TAG}")
    setel, lapor = [], []
    for _, pasangan in skenario.items():
        bagi = bagi_dua(pasangan)
        if bagi:
            setel += bagi[0]
            lapor += bagi[1]

    print(f"Set penyetelan n={len(setel)} | set pelaporan n={len(lapor)} (tidak beririsan)")
    print(f"Kriteria ditetapkan di muka: {KRITERIA}\n")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    hasil_setel, hasil_lapor, sifat, waktu, persist_map = {}, {}, {}, {}, {}
    for kode, ukuran, overlap, pemisah, jendela, ket in VARIAN:
        t0 = time.time()
        persist = TMP / kode.lower()
        n = indeks_ulang(ukuran, overlap, pemisah, jendela, persist)
        persist_map[kode] = persist
        import chromadb
        teks = chromadb.PersistentClient(path=str(persist)).get_collection(
            "diabetes_kb").get(include=["documents"])["documents"]
        pj = [len(tok.encode(t)) for t in teks]
        sifat[kode] = {
            "n_chunk": n, "token_maks": max(pj), "jendela": jendela,
            "n_melewati_jendela": sum(1 for x in pj if x > jendela),
            "persen_token_terbuang": round(
                100 * sum(max(0, x - jendela) for x in pj) / max(sum(pj), 1), 2),
        }
        hasil_setel[kode] = evaluasi(setel, persist, jendela)
        hasil_lapor[kode] = evaluasi(lapor, persist, jendela)
        waktu[kode] = round(time.time() - t0, 1)
        catat("jendela_token", kode, {**hasil_setel[kode], **sifat[kode]},
              catatan=f"{ket}, korpus={CORPUS_TAG}")
        s = sifat[kode]
        print(f"  {kode:6} jendela={jendela:4} {n:5} chunk | MRR lapor "
              f"{hasil_lapor[kode]['mrr']:.3f} hit@1 {hasil_lapor[kode]['hit@1(%)']:5.1f}% | "
              f"terbuang {s['persen_token_terbuang']:5.2f}% ({waktu[kode]:.0f} dtk)", flush=True)

    print("\nMemeriksa H2/H3 (apakah setelan jendela benar-benar terpasang) ...")
    vek = periksa_vektor(persist_map["W256"], persist_map["W512"])

    d = round(hasil_lapor["W512"]["mrr"] - hasil_lapor["W256"]["mrr"], 3)
    h1 = sifat["W512"]["n_melewati_jendela"] == 0
    # KOREKSI KRITERIA (16 Agustus 2026, sesudah jalan pertama).
    # Rumusan awal memakai kemiripan MAKS < 0,9999, yaitu menuntut SETIAP potongan
    # terpotong berubah. Itu salah rancang: potongan berukuran 257 token hanya
    # kehilangan satu token, sehingga vektornya memang nyaris tidak bergerak
    # (terukur 0,999966) tanpa berarti tombolnya tidak terpasang. Yang benar adalah
    # menilai RERATA. Koreksi ini menyangkut validasi alat, BUKAN arah hasil, dan
    # kedua rumusan tetap dicatat di keluaran supaya perubahannya terlihat.
    h2 = vek["H2_kemiripan_rerata_potongan_TERPOTONG"] is not None and \
        vek["H2_kemiripan_rerata_potongan_TERPOTONG"] < 0.99
    h2_rumusan_awal = vek["H2_kemiripan_maks_potongan_TERPOTONG"] is not None and \
        vek["H2_kemiripan_maks_potongan_TERPOTONG"] < 0.9999
    h3 = vek["H3_kemiripan_min_potongan_UTUH"] is not None and \
        vek["H3_kemiripan_min_potongan_UTUH"] > 0.9999

    if d > 0.02:
        putusan = ("Pemotongan senyap TERBUKTI merugikan retrieval: menghapusnya tanpa "
                   "menyentuh batas potongan menaikkan MRR. Menaikkan jendela layak diadopsi.")
    elif d < -0.02:
        putusan = ("Menghapus pemotongan justru MENURUNKAN MRR. Tidak boleh diadopsi "
                   "diam-diam; kemungkinan mutu representasi di token 257..512 memang "
                   "tidak tersetel pada model ini. Dilaporkan terbuka.")
    else:
        putusan = ("Pemotongan TIDAK terbukti berpengaruh pada metrik ini meskipun 8,23% "
                   "korpus tidak tervektor. Temuan ini dilaporkan apa adanya: integritas "
                   "korpus tetap alasan tersendiri untuk memperbaikinya, terpisah dari MRR.")

    hasil = {
        "percobaan": "T8 — pengaruh jendela token model embedding terhadap retrieval",
        "mengapa_sah": ("W256 dan W512 berpotongan IDENTIK (chunk_size, overlap, dan "
                        "pemisah sama), sehingga label classify_chunk() tidak bergeser. "
                        "Konfound yang membatalkan perbandingan T7 tidak berlaku di sini."),
        "dasar_angka_256": ("sentence_bert_config.json all-MiniLM-L6-v2; "
                            "config.json max_position_embeddings=512"),
        "kriteria": KRITERIA,
        "n_set_penyetelan": len(setel), "n_set_pelaporan": len(lapor),
        "varian": {k: {"chunk_size": u, "overlap": o, "pemisah_kalimat": p,
                       "jendela": j, "keterangan": ket}
                   for k, u, o, p, j, ket in VARIAN},
        "integritas": sifat,
        "waktu_dtk": waktu,
        "hasil_set_penyetelan": hasil_setel,
        "set_pelaporan": hasil_lapor,
        "pemeriksaan_vektor": vek,
        "dugaan": {
            "H1_nol_terpotong_pada_512": h1,
            "H2_vektor_BERUBAH_untuk_potongan_terpotong": h2,
            "H2_catatan_koreksi": (
                "Kriteria awal (maks < 0,9999) SALAH RANCANG dan menghasilkan GAGAL "
                "palsu: potongan 257 token hanya kehilangan satu token sehingga "
                "vektornya nyaris tidak bergerak. Dinilai ulang dengan RERATA. "
                "Koreksi menyangkut validasi alat, bukan arah hasil."),
            "H2_hasil_dengan_rumusan_awal": h2_rumusan_awal,
            "H3_vektor_IDENTIK_untuk_potongan_utuh": h3,
            "H4_arah_MRR": "tidak didugakan; kriteria 0,02 ditetapkan di muka",
        },
        "delta_mrr_W512_minus_W256": d,
        "putusan": putusan,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== SET PELAPORAN (n={len(lapor)}) ===")
    for k in hasil_lapor:
        print(f"  {k:6}: MRR {hasil_lapor[k]['mrr']:.3f}  hit@1 {hasil_lapor[k]['hit@1(%)']:5.1f}%"
              f"  terbuang {sifat[k]['persen_token_terbuang']:5.2f}%")
    print(f"\nH1 nol terpotong pada 512        : {'LOLOS' if h1 else 'GAGAL'}")
    print(f"H2 vektor berubah (yang terpotong): {'LOLOS' if h2 else 'GAGAL'} "
          f"(kemiripan maks {vek['H2_kemiripan_maks_potongan_TERPOTONG']})")
    print(f"H3 vektor identik (yang utuh)     : {'LOLOS' if h3 else 'GAGAL'} "
          f"(kemiripan min {vek['H3_kemiripan_min_potongan_UTUH']})")
    print(f"\ndelta MRR W512-W256 = {d:+.3f}\n{putusan}")
    print(f"\nDisimpan ke {OUT}")


if __name__ == "__main__":
    main()
