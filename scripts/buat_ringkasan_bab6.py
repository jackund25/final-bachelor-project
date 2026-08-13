"""T5.2 — kumpulkan seluruh angka yang akan masuk Bab VI ke SATU berkas.

Bab VI akan mengutip puluhan angka yang tersebar di belasan berkas JSON, sebagian di
antaranya ditulis pada tanggal berbeda oleh skrip berbeda. Risiko yang nyata, dan sudah
pernah terjadi dua kali dalam proyek ini, adalah mengutip angka dari berkas yang
KEDALUWARSA tanpa menyadarinya: faktor konformal 3,3 yang mengkalibrasi model yang sudah
tidak ada, dan `crossfold.json` 5 Agustus yang duduk berdampingan dengan berkas hari itu
sambil terlihat sah.

Skrip ini karena itu tidak sekadar menyalin angka. Untuk setiap sumber ia mencatat:

  * **mtime berkas**, sehingga berkas yang lebih tua daripada commit terakhir yang
    mengubah skrip penghasilnya dapat terlihat;
  * **status**: ADA, TIDAK ADA, atau RUSAK — tidak pernah dihilangkan diam-diam. Butir
    yang hilang dari ringkasan lebih berbahaya daripada butir yang ditandai kosong,
    karena yang pertama tidak menimbulkan pertanyaan apa pun.

Ringkasan juga memuat tanggal pembuatan dan **hash commit**, supaya setiap angka di Bab VI
dapat ditelusuri ke satu keadaan repositori yang pasti.

Keluaran: results/ringkasan_untuk_bab6.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ringkasan_untuk_bab6.json"

# (kunci, path relatif, daftar jalur-bertitik yang diambil). Jalur kosong = seluruh isi.
SUMBER: list[tuple[str, str, list[str]]] = [
    ("prediksi.rf_vs_lstm_h6", "results/eval_prediksi/crossval_rf_vs_lstm_h6.json",
     ["rerata_lintas_fold", "biaya", "uji", "k_fold"]),
    ("prediksi.rf_vs_lstm_h12", "results/eval_prediksi/crossval_rf_vs_lstm_h12.json",
     ["rerata_lintas_fold", "biaya", "uji", "k_fold"]),
    # Nama medan diperbaiki 11 Agustus 2026: skrip semula meminta `kriteria`,
    # `konfigurasi_terpilih`, dan `hasil_set_pelaporan`, yang tidak ada di berkas. Tiga
    # dari empat medan blok ini jatuh menjadi null tanpa terlihat, karena
    # `_medan_tidak_ditemukan` dicatat tetapi tidak pernah dicetak.
    ("prediksi.ukuran_rf_h6", "results/eval_prediksi/rf_ukuran_h6.json",
     ["kriteria_utama", "terpilih", "set_pelaporan", "n_konfigurasi_diuji",
      "PERINGATAN_PENAFSIRAN", "keputusan_produksi"]),
    ("prediksi.gradient_boosting_h6", "results/eval_prediksi/gradient_boosting_h6.json",
     ["rerata_lintas_fold", "biaya", "uji", "pemeriksaan_reproduksibilitas_RF",
      "hipoglikemia"]),
    ("prediksi.gradient_boosting_h12", "results/eval_prediksi/gradient_boosting_h12.json",
     ["rerata_lintas_fold", "biaya", "uji", "pemeriksaan_reproduksibilitas_RF",
      "hipoglikemia"]),
    ("prediksi.kontribusi_fitur", "results/eval_prediksi/feature_importance.json", []),
    ("prediksi.konformal_h6", "results/eval_prediksi/conformal_h6.json", ["levels"]),
    ("prediksi.konformal_h12", "results/eval_prediksi/conformal_h12.json", ["levels"]),
    ("prediksi.konformal_per_rentang_h6",
     "results/eval_prediksi/conformal_per_rentang_h6.json",
     ["agregat", "per_rentang", "hipotesis"]),
    ("prediksi.konformal_per_rentang_h12",
     "results/eval_prediksi/conformal_per_rentang_h12.json",
     ["agregat", "per_rentang", "hipotesis"]),
    ("prediksi.sensitivitas_tau_h6", "results/eval_prediksi/sensitivitas_tau_h6.json",
     ["per_sumbu", "rentang_rmse_persen_thd_produksi",
      "n_konfigurasi_berbeda_signifikan", "hasil_set_pelaporan", "rancangan",
      "verdik_kepekaan", "prapendaftaran_hasil", "susulan_PASCA_HOC"]),
    ("prediksi.hipoglikemia_h6", "results/eval_prediksi/hipoglikemia_h6.json",
     ["basis_kejadian", "rerata_lintas_fold", "uji_berpasangan_tingkat_fold",
      "terlewat_ke_rentang_target", "hipo_berat_terlewat"]),
    ("prediksi.hipoglikemia_h12", "results/eval_prediksi/hipoglikemia_h12.json",
     ["basis_kejadian", "rerata_lintas_fold", "uji_berpasangan_tingkat_fold"]),
    ("prediksi.pengklasifikasi_kondisi",
     "results/eval_prediksi/condition_classifier.json", []),
    # DIPERBAIKI 11 Agustus 2026 (C1/C2). Skrip semula menunjuk direktori `_kb12`, yang
    # crossfold.json-nya bertanggal 5 Agustus — era pra-A1, konfigurasi lambda 0,5 dan
    # bukan yang diadopsi. Ini persis kelas kesalahan yang docstring skrip ini sendiri
    # tulis sebagai alasan keberadaannya. Konfigurasi produksi (lambda 0,0) ada di
    # `_kb12_final`. Berkas `_kb12_sym` TETAP DISIMPAN sebagai jejak lambda 0,5.
    # Medan juga salah nama: `rerata_lintas_fold` tidak ada, yang benar
    # `ringkasan_lintas_fold`.
    ("retrieval.crossfold", "results/retrieval_realcases_kb12_final/crossfold.json",
     ["ringkasan_lintas_fold", "n_fold", "top_k", "catatan"]),
    ("retrieval.susunan_kueri", "results/eval_rag/susunan_kueri.json",
     ["hasil_set_penyetelan", "pemenang_set_penyetelan", "hasil_set_pelaporan",
      "protokol_bagian_c", "keterbatasan_menentukan",
      # Tabel peringkat T3.1 TIDAK BOLEH dikutip tanpa peringatan kontaminasinya.
      "KONTAMINASI_KOSAKATA_PELABEL", "PERBANDINGAN_KEBAL_KONTAMINASI",
      "PRAPENDAFTARAN_kondisi_dulu"]),
    ("retrieval.jangkauan_varian", "results/eval_rag/jangkauan_varian.json",
     ["A_jangkauan", "A_pertukaran_peringkat_lawan_jangkauan",
      "B_peringkat_bentuk_aplikasi", "C_kebal_kontaminasi", "D_susulan_PASCA_HOC"]),
    ("retrieval.konsentrasi", "results/eval_rag/konsentrasi_penelusuran.json",
     ["hasil_per_lambda", "mekanisme", "perbandingan_lambda", "yang_BELUM_dibuktikan"]),
    ("retrieval.horizon", "results/eval_rag/horizon_retrieval.json",
     ["hasil_set_penyetelan", "uji_berpasangan_set_penyetelan",
      "pemenang_dapat_dicapai", "hasil_set_pelaporan", "rumusan_yang_ditolak"]),
    ("retrieval.distribusi_token", "results/eval_rag/distribusi_token.json",
     ["batas_token_max_seq_length", "hasil", "catatan_model_max_length"]),
    ("retrieval.verifikasi_pelabel", "results/eval_rag/verifikasi_relevansi.json",
     ["n_dinilai", "kesepakatan_mentah_persen", "cohen_kappa", "tafsir_kappa",
      "per_kelas", "aturan_pelaporan"]),
    ("generasi.ragas", "results/ragas/summary.json", []),
    ("generasi.contoh_kasus", "results/ragas/contoh_kasus_bab6.json", []),
    ("sistem.penyetelan", "results/tuning_log.json", []),
]


def verifikasi_provenans(data: dict, rel: str) -> dict:
    """Periksa apakah berkas crossfold benar-benar dari konfigurasi produksi.

    Diminta: gagalkan skrip bila JSON tidak memuat lambda_mult 0.0, chunk_size 900, dan
    top_k 5. Kenyataannya **crossfold.json hanya merekam top_k**; `lambda_mult` dan
    `chunk_size` tidak pernah ditulis skrip penghasilnya. Assert literal karena itu akan
    selalu gagal dan tidak berguna.

    Yang dikerjakan sebagai gantinya, dan lebih jujur: pisahkan tiga keadaan —
    TERVERIFIKASI, TIDAK COCOK, dan TIDAK TERCATAT. Yang tidak cocok mematikan skrip.
    Yang tidak tercatat dilaporkan menonjol sebagai **cacat provenans**, karena artinya
    artefak itu tidak dapat menyatakan konfigurasinya sendiri dan satu-satunya buktinya
    adalah NAMA DIREKTORI — persis keadaan yang membuat berkas `_kb12` 5 Agustus nyaris
    terkutip sebagai hasil produksi.
    """
    harap = {"lambda_mult": 0.0, "chunk_size": 900, "top_k": 5}
    hasil = {"berkas": rel, "terverifikasi": {}, "tidak_cocok": {}, "tidak_tercatat": []}
    for k, v in harap.items():
        if k in data:
            if data[k] == v:
                hasil["terverifikasi"][k] = data[k]
            else:
                hasil["tidak_cocok"][k] = {"diharapkan": v, "terbaca": data[k]}
        else:
            hasil["tidak_tercatat"].append(k)
    hasil["catatan_cacah_provenans"] = (
        "Medan yang TIDAK TERCATAT tidak dapat diverifikasi dari artefak. Bukti bahwa "
        "berkas ini berasal dari lambda_mult 0,0 hanyalah nama direktorinya (_kb12_final) "
        "dan catatan journey part-5. Memperbaikinya menuntut skrip crossfold menuliskan "
        "konfigurasi efektifnya, lalu menjalankan ulang 2 jam 3 menit — di luar lingkup "
        "sesi ini, masuk daftar pekerjaan menunggu.")
    return hasil


def status_git_sumber(rel_paths: list[str]) -> dict:
    """Tandai sumber yang TERMODIFIKASI atau BELUM TERLACAK relatif terhadap git.

    Penjagaan atas kelas cacat yang ditemukan 13 Agustus 2026: `run_ragas.py` menimpa
    `summary.json` dan `per_sample.csv` diam-diam, bahkan menghilangkan satu medan. Audit
    menemukan **33 dari 41** skrip penulis di `scripts/` melakukan hal yang sama.

    Merombak ketiga puluh tiga skrip itu di luar lingkup. Penjagaan termurah yang
    benar-benar melindungi ada DI SINI: berkas hasil dilacak git, sehingga yang tertimpa
    muncul sebagai ``M`` pada ``git status`` — dan T5.2 satu-satunya tempat yang membaca
    keduapuluh empat sumber sekaligus.

    Sumber ``M`` belum tentu salah; ia hanya belum dicommit, sehingga angkanya tidak
    tertelusur ke commit yang dicatat ringkasan ini. Itu tepat yang perlu diperingatkan.
    """
    try:
        keluar = subprocess.run(["git", "status", "--porcelain", "--"] + list(rel_paths),
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return {}
    tanda = {}
    for baris in (keluar.stdout or "").splitlines():
        if len(baris) > 3:
            tanda[baris[3:].strip().strip('"')] = baris[:2].strip()
    return tanda


def ambil(d, jalur: str):
    v = d
    for k in jalur.split("."):
        if not isinstance(v, dict) or k not in v:
            return None
        v = v[k]
    return v


def git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=15).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    isi, hilang, rusak, medan_hilang = {}, [], [], []
    provenans = None
    tanda_git = status_git_sumber([rel for _, rel, _ in SUMBER])
    tak_tercommit = []
    for kunci, rel, jalur in SUMBER:
        p = ROOT / rel
        if not p.exists():
            # T2.2 belum diisi manusia. Ia TIDAK ditandai TIDAK ADA seperti berkas yang
            # sekadar belum dijalankan, karena sebabnya berbeda dan konsekuensinya lebih
            # besar: ia mengunci kualifikasi K1 atas SELURUH angka penelusuran.
            if kunci == "retrieval.verifikasi_pelabel":
                isi[kunci] = {
                    "_status": "MENUNGGU PENILAIAN MANUSIA",
                    "_berkas": rel,
                    "_berkas_untuk_diisi": "evaluation/verifikasi_relevansi.csv",
                    "_n_pasangan": 40,
                    "_kolom": "penilaian_manusia",
                    "_nilai_sah": ["hipoglikemia", "hiperglikemia", "normal", "lain"],
                    "_kunci_jangan_dibuka": "evaluation/verifikasi_relevansi_KUNCI.json",
                    "_setelah_diisi": "python scripts/verifikasi_relevansi.py --nilai",
                    "_PERINGATAN": (
                        "Selama kolom penilaian_manusia kosong, T3.1, T3.2, dan T3.3 TIDAK "
                        "BOLEH dinyatakan mantap. Seluruh metrik penelusuran (Hit@k, MRR, "
                        "nDCG@5) memakai relevansi dari classify_chunk(), pelabel kata "
                        "kunci — keterbatasan K1. Kesepakatan yang diukur T2.2 adalah "
                        "BATAS ATAS seberapa jauh angka-angka itu mencerminkan relevansi "
                        "sebenarnya."),
                }
                continue
            isi[kunci] = {"_status": "TIDAK ADA", "_berkas": rel}
            hilang.append((kunci, rel))
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            isi[kunci] = {"_status": "RUSAK", "_berkas": rel, "_galat": str(exc)}
            rusak.append((kunci, rel))
            continue
        mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        blok = {"_status": "ADA", "_berkas": rel, "_diubah": mtime}
        g = tanda_git.get(rel)
        if g:
            blok["_git"] = g
            blok["_PERINGATAN_GIT"] = (
                "Berkas TERMODIFIKASI atau BELUM TERLACAK git, sehingga angkanya TIDAK "
                "tertelusur ke commit yang dicatat ringkasan ini. Bila ini akibat skrip "
                "yang menimpa berkas lama, periksa `git diff` sebelum mengutipnya.")
            tak_tercommit.append((kunci, rel, g))
        if kunci == "retrieval.crossfold":
            provenans = verifikasi_provenans(data, rel)
            blok["_verifikasi_provenans"] = provenans
        if jalur:
            for j in jalur:
                v = ambil(data, j)
                blok[j] = v
                if v is None:
                    blok.setdefault("_medan_tidak_ditemukan", []).append(j)
                    medan_hilang.append((kunci, j, rel))
        else:
            blok["isi"] = data
        isi[kunci] = blok

    ringkas = {
        "judul": "Ringkasan angka untuk Bab VI",
        "dibuat": datetime.now().isoformat(timespec="seconds"),
        "dibuat_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git("rev-parse", "HEAD"),
        "commit_pendek": git("rev-parse", "--short", "HEAD"),
        "cabang": git("rev-parse", "--abbrev-ref", "HEAD"),
        "pohon_kerja_bersih": (git("status", "--porcelain") or "") == "",
        "cara_pakai": (
            "Setiap blok memuat _status dan _diubah. Sebelum mengutip angka mana pun, "
            "periksa _diubah terhadap tanggal commit yang terakhir mengubah skrip "
            "penghasilnya. Angka yang lebih tua daripada perubahan skripnya adalah "
            "angka yang dihasilkan kode yang sudah tidak ada."),
        "peringatan_riwayat": (
            "Dua kali dalam proyek ini angka kedaluwarsa nyaris terpakai: faktor "
            "konformal 3,3 yang mengkalibrasi model pra-Tugas 5, dan crossfold.json "
            "5 Agustus yang berdampingan dengan berkas baru sambil terlihat sah. "
            "Medan _diubah ada untuk itu."),
        "n_sumber": len(SUMBER),
        "n_ada": len(SUMBER) - len(hilang) - len(rusak)
                 - sum(1 for v in isi.values()
                       if v.get("_status") == "MENUNGGU PENILAIAN MANUSIA"),
        "n_tidak_ada": len(hilang),
        "n_rusak": len(rusak),
        "n_menunggu_manusia": sum(1 for v in isi.values()
                                  if v.get("_status") == "MENUNGGU PENILAIAN MANUSIA"),
        "daftar_tidak_ada": [{"kunci": k, "berkas": r} for k, r in hilang],
        "daftar_rusak": [{"kunci": k, "berkas": r} for k, r in rusak],
        "daftar_medan_tidak_ditemukan": [{"kunci": k, "medan": j, "berkas": r}
                                         for k, j, r in medan_hilang],
        "verifikasi_provenans_crossfold": provenans,
        "sumber_belum_tercommit": [{"kunci": k, "berkas": r, "git": g}
                                   for k, r, g in tak_tercommit],
        "keterbatasan": "docs/DAFTAR_KETERBATASAN.md",
        "data": isi,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ringkas, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"T5.2 — ringkasan Bab VI")
    print(f"  commit  : {ringkas['commit_pendek']} ({ringkas['cabang']})"
          f"{'' if ringkas['pohon_kerja_bersih'] else '  [POHON KERJA BELUM BERSIH]'}")
    print(f"  sumber  : {ringkas['n_ada']}/{ringkas['n_sumber']} ada, "
          f"{ringkas['n_tidak_ada']} tidak ada, {ringkas['n_rusak']} rusak")
    if hilang:
        print("\n  BELUM ADA (ditandai di berkas, bukan dihilangkan):")
        for k, r in hilang:
            print(f"    - {k:<38} {r}")
    if rusak:
        print("\n  RUSAK:")
        for k, r in rusak:
            print(f"    - {k:<38} {r}")
    kadaluwarsa = [(k, v["_diubah"]) for k, v in isi.items()
                   if v.get("_status") == "ADA" and v["_diubah"] < "2026-08-01"]
    if kadaluwarsa:
        print("\n  PERIKSA TANGGAL (diubah sebelum 1 Agustus 2026):")
        for k, t in sorted(kadaluwarsa, key=lambda x: x[1]):
            print(f"    - {k:<38} {t}")

    menunggu = [(k, v) for k, v in isi.items()
                if v.get("_status") == "MENUNGGU PENILAIAN MANUSIA"]
    for k, v in menunggu:
        print(f"\n  {'=' * 68}")
        print(f"  MENUNGGU PENILAIAN MANUSIA: {k}")
        print(f"  {'=' * 68}")
        print(f"    isi {v['_n_pasangan']} baris kolom `{v['_kolom']}` pada "
              f"{v['_berkas_untuk_diisi']}")
        print(f"    nilai sah   : {' | '.join(v['_nilai_sah'])}")
        print(f"    JANGAN buka : {v['_kunci_jangan_dibuka']} sebelum selesai")
        print(f"    lalu        : {v['_setelah_diisi']}")
        print(f"    PERINGATAN  : selama kosong, T3.1, T3.2, dan T3.3 TIDAK BOLEH")
        print(f"                  dinyatakan mantap (keterbatasan K1)")

    if provenans:
        pv = provenans
        if pv["tidak_cocok"]:
            print(f"\n  {'!' * 68}")
            print(f"  PROVENANS TIDAK COCOK pada {pv['berkas']}:")
            for k, d in pv["tidak_cocok"].items():
                print(f"    - {k}: diharapkan {d['diharapkan']}, terbaca {d['terbaca']}")
            print(f"  Berkas ini BUKAN dari konfigurasi produksi. Angkanya tidak boleh")
            print(f"  dikutip. Skrip dihentikan.")
            print(f"  {'!' * 68}")
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(ringkas, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            return 2
        if pv["tidak_tercatat"]:
            print(f"\n  CACAT PROVENANS pada {pv['berkas']}:")
            print(f"    terverifikasi  : {pv['terverifikasi']}")
            print(f"    TIDAK TERCATAT : {pv['tidak_tercatat']}")
            print(f"    Artefak tidak dapat menyatakan konfigurasinya sendiri; bukti satu-")
            print(f"    satunya adalah NAMA DIREKTORI. Ini keadaan yang membuat berkas")
            print(f"    _kb12 5 Agustus nyaris terkutip sebagai hasil produksi.")

    if tak_tercommit:
        print(f"\n  {'!' * 68}")
        print(f"  {len(tak_tercommit)} SUMBER BELUM TERCOMMIT — angkanya tidak tertelusur "
              f"ke commit di atas:")
        for k, r, g in tak_tercommit:
            print(f"    [{g:<2}] {k:<34} {r}")
        print("  Berkas hasil dilacak git justru agar penimpaan diam-diam terlihat;")
        print("  33 dari 41 skrip penulis menimpa tanpa penjagaan (audit 13 Agu 2026).")
        print("  Periksa `git diff` sebelum mengutip angkanya.")
        print(f"  {'!' * 68}")

    if medan_hilang:
        print(f"\n  {'!' * 68}")
        print(f"  MEDAN TIDAK DITEMUKAN — {len(medan_hilang)} medan jatuh menjadi null:")
        for k, j, r in medan_hilang:
            print(f"    - {k:<36} medan `{j}`  ({r})")
        print(f"  Nama medan yang salah membuat angka HILANG TANPA TERLIHAT. Perbaiki")
        print(f"  daftar SUMBER, jangan biarkan ringkasan tampak sehat sambil kosong.")
        print(f"  {'!' * 68}")

    print(f"\nDisimpan ke {OUT}")
    if medan_hilang:
        print("Keluar dengan kode 1 karena ada medan yang tidak ditemukan.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
