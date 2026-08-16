"""Menjaga agar dua salinan conformal_q tidak menyimpang.

scripts/eval_cakupan_conformal.py MENYALIN conformal_q dari
scripts/conformal_calibration.py alih-alih mengimpornya, karena modul itu
mengawali dengan `import torch` untuk menata urutan DLL dan torch kadang gagal
memuat c10.dll pada mesin ini.

Salinan yang menyimpang diam-diam akan membuat `q` yang dipakai runtime berbeda
dari `q` yang dipakai menaksir cakupan — persis jenis ketidakselarasan senyap yang
menjadi pokok T12. Berkas ini menutup celah itu.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _ambil_sumber(path: Path, nama: str) -> str:
    """Ambil sumber satu fungsi TANPA mengimpor modulnya.

    Diambil lewat AST justru karena mengimpor conformal_calibration.py akan
    menarik torch, yang merupakan sebab pemisahan ini ada.
    """
    pohon = ast.parse(path.read_text(encoding="utf-8"))
    for simpul in ast.walk(pohon):
        if isinstance(simpul, ast.FunctionDef) and simpul.name == nama:
            # Docstring dibuang: keduanya memang sengaja berbeda penjelasannya.
            badan = [n for n in simpul.body
                     if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                             and isinstance(n.value.value, str))]
            return "\n".join(ast.dump(n) for n in badan)
    raise AssertionError(f"fungsi {nama} tidak ditemukan di {path}")


def test_dua_salinan_conformal_q_identik():
    """Badan fungsinya wajib sama persis, hanya docstring-nya yang boleh berbeda."""
    a = _ambil_sumber(ROOT / "scripts/conformal_calibration.py", "conformal_q")
    b = _ambil_sumber(ROOT / "scripts/eval_cakupan_conformal.py", "conformal_q")
    assert a == b, (
        "conformal_q pada eval_cakupan_conformal.py sudah menyimpang dari "
        "conformal_calibration.py. Samakan keduanya, atau hapus salinannya."
    )


def test_koreksi_sampel_hingga_lebih_konservatif_daripada_kuantil_biasa():
    """Koreksi (n+1)/n itulah yang membuat jaminan conformal berlaku.

    Tanpa koreksi ini q terlalu kecil dan cakupannya meleset ke bawah — kegagalan
    yang tidak terlihat dari angka mana pun kecuali cakupan diukur pada himpunan
    yang benar-benar terpisah.
    """
    pytest.importorskip("numpy")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_ecc", ROOT / "scripts/eval_cakupan_conformal.py")
    # Modul ini mengimpor sklearn dan src.*; cukup ambil fungsinya lewat exec
    # terbatas agar tes tetap ringan.
    sumber = (ROOT / "scripts/eval_cakupan_conformal.py").read_text(encoding="utf-8")
    mulai = sumber.index("def conformal_q")
    akhir = sumber.index("\n\n", sumber.index("method=\"higher\"", mulai))
    ruang = {"np": np}
    exec(sumber[mulai:akhir], ruang)  # noqa: S102 — sumber milik repo sendiri
    conformal_q = ruang["conformal_q"]

    skor = np.arange(100, dtype=float)
    q = conformal_q(skor, alpha=0.05)

    assert q >= float(np.quantile(skor, 0.95, method="higher"))
    assert conformal_q(skor, 0.05) >= conformal_q(skor, 0.10)


def test_q_tidak_melampaui_skor_terbesar():
    """Kuantil dengan level dipangkas ke 1,0 tidak boleh melampaui data."""
    sumber = (ROOT / "scripts/eval_cakupan_conformal.py").read_text(encoding="utf-8")
    mulai = sumber.index("def conformal_q")
    akhir = sumber.index("\n\n", sumber.index("method=\"higher\"", mulai))
    ruang = {"np": np}
    exec(sumber[mulai:akhir], ruang)  # noqa: S102
    conformal_q = ruang["conformal_q"]

    # n kecil membuat (n+1)/n * (1-alpha) melampaui 1 dan harus dipangkas.
    skor = np.array([1.0, 2.0, 3.0])
    assert conformal_q(skor, 0.05) == 3.0
