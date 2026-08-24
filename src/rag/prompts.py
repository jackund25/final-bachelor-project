"""Prompt templates and context formatting utilities for diabetes RAG."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Struktur keluaran mengikuti SBAR (Situation, Background, Assessment,
# Recommendation) — Muller M, Jurgens J, Redaelli M, dkk. "Impact of the
# communication and patient hand-off tool SBAR on patient safety: a systematic
# review." BMJ Open. 2018;8(8):e022202. doi:10.1136/bmjopen-2018-022202
# (akses terbuka, CC BY-NC 4.0).
#
# Rujukan ini dipilih menggantikan Haig dkk. (2006), makalah asal SBAR, karena
# teks penuhnya tidak dapat diakses. Tinjauan sistematis juga lebih tinggi
# tingkat buktinya daripada studi kasus, dan ia memuat definisi tiap komponen:
# S = apa yang sedang terjadi pada pasien; B = latar/konteks pasien; A = apa
# masalahnya; R = langkah berikutnya dalam tata laksana. Kelima bagian di bawah
# diturunkan langsung dari definisi itu.
#
# PEMBAGIAN ISI MENGIKUTI TABEL 1 MAKALAH, bukan tebakan. Di sana S hanya
# pernyataan singkat masalah; data kuantitatif (tanda vital, hasil lab, keadaan
# terkini) justru milik A, bersama kesan klinis atasnya; B memuat konteks dan
# riwayat pasien; R memuat saran beserta apa yang diperlukan untuk menangani
# masalahnya. Karena itu angka glukosa ditempatkan di Penilaian, BUKAN di
# Situasi — penempatan yang keliru akan terlihat oleh pembaca yang membuka
# Tabel 1.
#
# MENGAPA SBAR, BUKAN STRUKTUR KARANGAN SENDIRI. Nilainya bukan pada kerapian
# melainkan pada urutan tetap yang SUDAH ADA di kepala dokter — tinjauan itu
# menyebut pengirim dan penerima berbagi model mental yang sama. Dokter tahu apa
# yang datang berikutnya, sehingga beban membaca turun dan bagian yang hilang
# langsung terasa. Advisory berbentuk paragraf datar memaksa dokter mengekstraksi
# struktur itu sendiri, dan itulah kerja yang seharusnya diambil alih sistem.
#
# BATAS KLAIM YANG BOLEH DIAMBIL. Tinjauan itu menemukan bukti MODERATE saja:
# 2 dari 11 studi bermutu kuat/sedang, 8 sisanya berdesain before-after yang
# lemah. SBAR karena itu diadopsi di sini atas sifat PENSTRUKTURAN-nya, BUKAN
# sebagai klaim bahwa sistem ini meningkatkan keselamatan pasien.
#
# SBAR juga membenarkan pembagian peran yang dipegang sistem ini: huruf R adalah
# rekomendasi dari pihak yang MELAPOR kepada pihak yang BERWENANG MEMUTUSKAN.
# Sistem menempati posisi pelapor, dokter memegang keputusan (KNF-06).
#
# ADAPTASI, BUKAN PENERAPAN LANGSUNG. SBAR dirancang untuk komunikasi
# antar-manusia (perawat -> dokter). Memakainya untuk keluaran mesin adalah
# adaptasi, dan itu dinyatakan terbuka di Bab IV.
#
# BAGIAN 5 BUKAN BAGIAN DARI SBAR. SBAR tidak punya slot untuk batas inferensi.
# Bagian itu tambahan, dibenarkan DECIDE-AI (Vasey dkk., Nat Med 2022;28:924-933)
# yang meminta sistem pendukung keputusan menyatakan batas kemampuannya pada
# evaluasi tahap awal. Ia juga satu-satunya bagian yang tidak dapat ditiru oleh
# aturan ambang if/else.
SYSTEM_PROMPT = """Anda adalah asisten klinis berbasis panduan medis Indonesia untuk mendukung keputusan dokter dalam penanganan diabetes.

Aturan:
1. Jawab berdasarkan konteks yang diberikan.
2. JANGAN menulis nomor halaman, nomor bab, nomor tabel, atau tautan.
   Rujuk sumber HANYA dengan penanda [S1], [S2], ... sesuai nomor blok konteks.
3. Jika blok konteks kosong, nyatakan secara eksplisit bahwa tidak ada rujukan
   panduan yang relevan pada knowledge base, dan JANGAN mengarang rujukan.
4. Jika konteks tidak cukup, katakan informasi belum tersedia pada knowledge base saat ini.
5. Untuk kondisi berisiko tinggi, sarankan evaluasi dokter segera.
6. Jika pertanyaan meminta nilai, ambang, dosis, rentang, atau parameter
   spesifik dan informasi tersebut tersedia secara eksplisit dalam konteks,
   jawab dengan nilai yang tercantum dalam konteks tersebut. Jangan menggantinya
   dengan jawaban generik. Jika terdapat beberapa nilai yang berbeda, jelaskan
   perbedaannya berdasarkan konteks dan jangan memilih angka tanpa dasar.
7. Gunakan Bahasa Indonesia yang ringkas, jelas, dan actionable.
8. Penanda [S...] hanya boleh dilekatkan pada pernyataan yang isinya BENAR-BENAR
   ada pada blok konteks bernomor itu. Bila suatu angka, ambang, klasifikasi, atau
   tata laksana TIDAK ada pada blok manapun, JANGAN memberinya penanda: tulis
   tanpa penanda dan nyatakan bahwa panduan yang terambil tidak memuatnya, atau
   hilangkan pernyataan itu. Menempelkan penanda pada pengetahuan yang berasal
   dari luar konteks adalah sitasi palsu dan merusak keterlacakan bukti — dokter
   akan membuka halaman yang ditunjuk dan tidak menemukan isinya di sana.
9. Tulis notasi matematika sebagai teks biasa. Pakai "kurang dari atau sama
   dengan 70 mg/dL" atau "<= 70 mg/dL". JANGAN memakai LaTeX, tanda dolar, atau
   markup rumus — tampilan tidak merendernya dan dokter akan melihat kodenya
   mentah.

Bentuk jawaban:
Susun jawaban dalam lima bagian berikut, dengan judul persis seperti tertulis,
berurutan, dan tidak ada bagian yang dilewati.

**Situasi**
Pernyataan singkat masalahnya: kondisi apa yang diprediksi terjadi pada horizon
prediksi. Satu sampai dua kalimat, TANPA merinci angka — angka masuk ke bagian
Penilaian.

**Latar**
Konteks pasien yang relevan: modalitas pengukuran (CGM atau finger-stick),
riwayat yang tersedia, serta faktor yang menjelaskan mengapa kondisi itu
diperkirakan terjadi — insulin aktif, karbohidrat aktif, aktivitas. Hanya yang
tersedia pada data pasien; jangan menambah faktor yang tidak ada datanya.

**Penilaian**
Data kuantitatif keadaan terkini dan terprediksi — glukosa saat ini, glukosa
prediksi, interval prediksi bila ada, laju perubahan — DIIKUTI kesan klinis atas
data itu: apa artinya dan seberapa gawat menurut panduan. Bagian ini wajib
merujuk konteks dengan penanda [S...].

**Rekomendasi**
a. Tindakan, disusun BERJENJANG menurut derajat kondisi — bukan satu saran tunggal.
b. Parameter yang dipantau BESERTA jadwalnya (berapa sering, sampai kapan).
c. Kriteria eskalasi atau rujukan: ambang atau keadaan yang menuntut tindakan
   lebih lanjut.
Setiap tindakan, angka, dan ambang pada bagian ini WAJIB diikuti penanda [S...]
yang menjadi dasarnya, ditempelkan pada tindakannya, bukan dikumpulkan di akhir.

**Yang tidak dapat disimpulkan dari data ini**
Batas yang jujur: apa yang tidak dapat dinilai dari data yang tersedia, dan
ketidakpastian yang melekat pada prediksi. Contoh yang relevan: horizon prediksi
yang panjang, modalitas pengukuran yang jarang, interval prediksi yang lebar,
atau parameter klinis yang tidak ada pada data.

Aturan pengisian bagian:
- Bila konteks tidak memuat bahan untuk suatu bagian, tulis pada bagian itu bahwa
  panduan yang terambil tidak memuatnya. JANGAN mengisinya dengan nasihat umum,
  dan JANGAN mengarang ambang atau dosis.
- JANGAN menghapus judul bagian mana pun, termasuk ketika isinya menyatakan
  ketiadaan informasi.
- JANGAN menutup jawaban dengan disclaimer, catatan penutup, atau kalimat yang
  menyatakan keputusan ada pada dokter. Sistem menambahkan catatan itu sendiri
  dengan kata-kata baku; menuliskannya sendiri membuat disclaimer muncul DUA KALI
  pada jawaban yang dibaca dokter.
"""


def format_context_with_citations(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a readable context block with source markers.

    Nomor halaman SENGAJA TIDAK dimasukkan ke blok konteks. Aturan pada system prompt
    saja hanyalah jaminan lunak; tidak memberikan angkanya sama sekali adalah jaminan
    keras — model tidak dapat menyalin nomor halaman yang tak pernah dilihatnya.
    Penomoran halaman ditangani lapisan UI dari metadata chunk (src/rag/citations.py).
    """
    if not retrieved_docs:
        return "(Tidak ada konteks dokumen yang ditemukan)"

    from .citations import document_title

    lines: List[str] = []
    for idx, row in enumerate(retrieved_docs, start=1):
        metadata = dict(row.get("metadata", {}))
        fallback_name = row.get("source", "Manual KB")
        judul = document_title(metadata, fallback_name)
        lembaga = metadata.get("lembaga") or metadata.get("sumber") or fallback_name
        tahun = metadata.get("tahun", "N/A")
        marker = f"[S{idx}] {judul} — {lembaga}, {tahun}"
        lines.append(f"{marker}\n{row.get('text', '').strip()}")

    return "\n\n".join(lines)


def build_question_payload(
    query: str,
    patient_state: Dict[str, Any],
    prediction: float,
    horizon_minutes: Optional[int] = None,
    risk_label: Optional[str] = None,
) -> str:
    """Build a compact clinician question payload.

    horizon_minutes WAJIB diteruskan pemanggil. Sebelumnya teks "Prediksi 1 jam"
    di-hardcode di sini, sehingga prompt memberi tahu LLM horizon 60 menit padahal
    bundle produksi memprediksi 30 menit — bertentangan pula dengan kueri retrieval
    dan dengan angka yang ditampilkan UI.
    """
    # Handle prediction dict or numeric
    if isinstance(prediction, dict):
        pred_glucose = prediction.get('glucose_pred', '?')
        pred_risk = prediction.get('risk_level', 'N/A')
    else:
        pred_glucose = f"{float(prediction):.1f}" if prediction else '?'
        # Label risiko diturunkan dari ambang tunggal bila pemanggil tidak memberikannya.
        # Sebelumnya nilai ini SELALU 'N/A' pada jalur produksi karena pipeline selalu
        # mengirim float, sehingga LLM tidak pernah menerima status risiko sama sekali.
        if risk_label is not None:
            pred_risk = risk_label
        else:
            from src.constants import classify_glucose_5zone, risk_label_id
            pred_risk = risk_label_id(classify_glucose_5zone(float(prediction)))

    # Support both key conventions
    gluc = patient_state.get('glucose', patient_state.get('current_glucose', 'N/A'))
    activity = patient_state.get('activity', patient_state.get('activity_level', 0))
    # `iob`/`cob` didahulukan — itu nama fitur pada jalur produksi. Membaca
    # 'insulin'/'carbs' lebih dulu membuat nilainya selalu 0 tanpa error.
    insulin = patient_state.get(
        'iob', patient_state.get('insulin_on_board', patient_state.get('insulin', 0)))
    carbs = patient_state.get(
        'cob', patient_state.get('carbs_on_board', patient_state.get('carbs', 0)))
    # Stres tidak termasuk engineered_features, jadi pada produksi ia tidak pernah ada.
    # Bila tidak ada, barisnya DIHILANGKAN — bukan diisi default 5 yang lalu dibaca
    # dokter sebagai skor yang benar-benar diukur.
    stress = patient_state.get('stress', patient_state.get('stress_level'))
    stress_txt = f"stress={stress}/10, " if stress is not None else ""

    horizon_txt = f"{horizon_minutes} menit" if horizon_minutes else "horizon prediksi"

    return (
        f"Pertanyaan klinisi: {query}\n"
        f"Data pasien: glukosa={gluc} mg/dL, "
        f"{stress_txt}"
        f"aktivitas={activity} (skor intensitas), "
        f"insulin_on_board={insulin} unit, "
        f"carbs_on_board={carbs} gram.\n"
        f"Prediksi {horizon_txt}: {pred_glucose} mg/dL, Risiko: {pred_risk}."
    )
