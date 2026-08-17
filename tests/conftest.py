import json
import sys
from pathlib import Path

import pytest

# Ensure project root is importable when running tests directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Potongan sintetis bermetadata lengkap, meniru bentuk fallback_chunks.json yang
# diekspor scripts/reingest_kb.py. Isinya sengaja pendek dan cukup untuk menguji
# jalur cadangan berbasis kata kunci.
_POTONGAN_UJI = [
    {
        "text": (
            "Hipoglikemia adalah kadar glukosa darah di bawah 70 mg/dL. Penanganan "
            "awalnya mengikuti aturan 15-15: berikan 15 gram karbohidrat kerja cepat, "
            "tunggu 15 menit, lalu periksa ulang kadar glukosa darah."
        ),
        "source": "KB-UJI_Pedoman-Sintetis.pdf",
        "metadata": {
            "kb_id": "KB-UJI", "source_id": "KB-UJI_p1",
            "nama_dokumen": "KB-UJI_Pedoman-Sintetis.pdf",
            "lembaga": "Uji", "tahun": 2026,
            "judul_lengkap": "Pedoman sintetis untuk pengujian",
            "halaman_pdf": 1, "halaman_cetak": 1, "halaman_cetak_valid": True,
        },
    },
    {
        "text": (
            "Hiperglikemia adalah kadar glukosa darah di atas 180 mg/dL. Pemantauan "
            "gula darah mandiri dilakukan lebih sering, asupan cairan ditingkatkan, "
            "dan keton diperiksa bila kadar glukosa melampaui 250 mg/dL."
        ),
        "source": "KB-UJI_Pedoman-Sintetis.pdf",
        "metadata": {
            "kb_id": "KB-UJI", "source_id": "KB-UJI_p2",
            "nama_dokumen": "KB-UJI_Pedoman-Sintetis.pdf",
            "lembaga": "Uji", "tahun": 2026,
            "judul_lengkap": "Pedoman sintetis untuk pengujian",
            "halaman_pdf": 2, "halaman_cetak": 2, "halaman_cetak_valid": True,
        },
    },
    {
        "text": (
            "Edukasi dasar pemantauan gula darah mencakup waktu pemeriksaan, cara "
            "mencatat hasil pada logbook, dan target glikemik yang disepakati bersama "
            "dokter. Pencatatan yang teratur memudahkan penilaian tren."
        ),
        "source": "KB-UJI_Pedoman-Sintetis.pdf",
        "metadata": {
            "kb_id": "KB-UJI", "source_id": "KB-UJI_p3",
            "nama_dokumen": "KB-UJI_Pedoman-Sintetis.pdf",
            "lembaga": "Uji", "tahun": 2026,
            "judul_lengkap": "Pedoman sintetis untuk pengujian",
            "halaman_pdf": 3, "halaman_cetak": 3, "halaman_cetak_valid": True,
        },
    },
]


@pytest.fixture
def kb_dir_uji(tmp_path):
    """Direktori basis pengetahuan berisi potongan cadangan sintetis.

    Tes yang membangun ``RAGPipeline`` dengan direktori kosong dahulu tetap berjalan
    karena ``muat_potongan_cadangan`` diam-diam jatuh ke ``data/knowledge_base``.
    Perilaku itu dicabut, sehingga hasilnya bergantung pada ada tidaknya indeks
    produksi di mesin yang menjalankan tes — dan itu membuat tes lulus atau gagal
    menurut keadaan di luar dirinya. Fixture ini membuat kebergantungan itu eksplisit.
    """
    (tmp_path / "fallback_chunks.json").write_text(
        json.dumps(_POTONGAN_UJI, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(tmp_path)
