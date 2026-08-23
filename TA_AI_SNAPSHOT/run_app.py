"""Launcher aplikasi Streamlit.

Di Windows, `torch` HARUS dimuat sebelum Streamlit (yang meng-import numpy/pyarrow
saat startup). Jika tidak, inisialisasi `c10.dll` gagal dengan WinError 1114.
Perintah `streamlit run app/streamlit_app.py` meng-import Streamlit lebih dulu,
sehingga `import torch` di dalam skrip selalu terlambat — karena itu pakai launcher ini.

Cara pakai:
    conda activate diabetes-ta
    python run_app.py
"""

import torch  # noqa: F401 — WAJIB pertama, sebelum Streamlit menyentuh numpy/pyarrow

import sys
from pathlib import Path

from streamlit.web import cli as stcli

if __name__ == "__main__":
    app = str(Path(__file__).parent / "app" / "streamlit_app.py")
    sys.argv = ["streamlit", "run", app]
    sys.exit(stcli.main())
