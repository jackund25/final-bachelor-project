# SPEC BAB IV — Dokumentasi Teknis Sistem (dari kode yang ada)

> Dokumen ini disusun **hanya dari pembacaan kode**, bukan dari dokumen desain.
> Setiap poin menyertakan path berkas dan nomor baris.
> Tanda **TIDAK DITEMUKAN** berarti hal tersebut tidak ada di dalam kode.
>
> **Versi 1** — pembacaan kode 2026-08-04, sebelum refaktor.
> **Versi 2** — diperbarui 2026-08-05 setelah refaktor tujuh tugas.

## Cara membaca revisi ini

Perubahan terhadap Versi 1 ditandai eksplisit dengan blockquote:

> **BERUBAH (Tugas N):** penjelasan singkat apa yang berbeda dari versi sebelumnya.

Bagian tanpa penanda berarti **tidak berubah** dari Versi 1.

### Ringkasan bagian yang berubah

| Bagian | Status | Penyebab |
|---|---|---|
| A. Parser | Sebagian berubah | Tugas 5 (segmentasi jeda) |
| B. Rekayasa fitur | Sebagian berubah | Tugas 5 (jendela tersaring) |
| C. Model prediksi | Sebagian berubah | Tugas 5 (RMSE baru) |
| D. Kalibrasi ketidakpastian | Tidak berubah | — |
| E. Klasifikasi kondisi | **Berubah** | Tugas 3 (ambang tunggal) |
| F. Pengindeksan korpus | **Berubah total** | Tugas 1 (korpus + metadata halaman) |
| G. Penelusuran & pembangkitan | **Berubah** | Tugas 1B, 4 (sitasi, skor, parameter LLM) |
| H. Struktur perangkat lunak | **Berubah** | Tugas 2 (hapus digital twin) |
| I. Konfigurasi | **Berubah total** | Tugas 2, 4, 5 (config otoritatif) |
| Lampiran temuan | **Berubah** | 12 dari 22 temuan terselesaikan |

Riwayat perubahan per tugas beserta commit-nya ada di `docs/journey.md`.

---

## A. PARSER DAN PENGOLAHAN DATA

### A.1 Urutan fungsi dari XML mentah sampai CSV

Seluruh jalur parser ada di [src/data/ohio_parser.py](../src/data/ohio_parser.py).

| # | Fungsi | Lokasi | Peran |
|---|--------|--------|-------|
| 1 | `process_ohio_dataset()` | [ohio_parser.py:210-230](../src/data/ohio_parser.py#L210-L230) | Entry point. Glob `**/[0-9]*-ws-*.xml` (baris 217), lalu memanggil dua jalur ekspor. |
| 2 | `_merge_and_write(parse_ohio_xml, ...)` | [ohio_parser.py:189-207](../src/data/ohio_parser.py#L189-L207) | Loop file, gabung, sort `[patient_id, timestamp]`, `to_csv`. |
| 3 | `parse_ohio_xml()` | [ohio_parser.py:153-168](../src/data/ohio_parser.py#L153-L168) | Parse 1 file → timeline CGM. |
| 4 | `_events_to_df(root, "glucose_level", "ts", ["value"])` | [ohio_parser.py:162](../src/data/ohio_parser.py#L162) | Ambil kanal CGM sebagai timeline dasar (`base`). |
| 5 | filter `cgm["glucose"] > 0` | [ohio_parser.py:165](../src/data/ohio_parser.py#L165) | Buang pembacaan ≤ 0. |
| 6 | `_build_feature_frame()` | [ohio_parser.py:113-150](../src/data/ohio_parser.py#L113-L150) | Ekstrak semua kanal event, align ke grid CGM, rakit DataFrame. |
| 6a | `_events_to_df()` × 8 kanal | [ohio_parser.py:118-125](../src/data/ohio_parser.py#L118-L125) | meal, bolus, basal, exercise, stressors, sleep, work, illness. |
| 6b | `_align_sum()` | [ohio_parser.py:64-78](../src/data/ohio_parser.py#L64-L78) | carbs, bolus_dose, activity. |
| 6c | `_basal_rate_stepwise()` | [ohio_parser.py:98-110](../src/data/ohio_parser.py#L98-L110) | basal_rate (step function). |
| 6d | `_align_presence()` | [ohio_parser.py:81-95](../src/data/ohio_parser.py#L81-L95) | stress, sleep, work, illness (flag 0/1). |
| 7 | `_parse_ts()` | [ohio_parser.py:31-37](../src/data/ohio_parser.py#L31-L37) | Format `"%d-%m-%Y %H:%M:%S"` (baris 26). |

Jalur kedua (paralel, bukan lanjutan): `parse_ohio_fingerstick()` [ohio_parser.py:171-186](../src/data/ohio_parser.py#L171-L186), memakai kanal `finger_stick` sebagai timeline dan memanggil `_build_feature_frame()` yang sama.

**Keluaran:**
- `data/raw/ohio_t1dm_merged.csv` — timeline CGM 5 menit (dataset training)
- `data/raw/ohio_t1dm_smbg.csv` — timeline finger_stick (skenario SMBG)

**Catatan penting:** parser ini **tidak** memanggil `DataPreprocessor`. Pembersihan nilai kosong dan rekayasa fitur terjadi terpisah, pada saat pelatihan ([rf_model.py:118](../src/models/rf_model.py#L118) dan [rf_model.py:130](../src/models/rf_model.py#L130)).

**Kolom CSV keluaran** (verifikasi langsung dari `ohio_t1dm_merged.csv`):
```
timestamp, patient_id, glucose, carbs, insulin, bolus_dose, basal_rate,
activity, stress, sleep, work, illness, meal_type, source
```

### A.2 Kanal XML yang dibaca dan yang diabaikan

Hasil enumerasi langsung isi `data/raw/OhioT1DM/2018/train/559-ws-training.xml` dibandingkan dengan kode di [ohio_parser.py:118-125](../src/data/ohio_parser.py#L118-L125) dan [ohio_parser.py:162](../src/data/ohio_parser.py#L162), [ohio_parser.py:180](../src/data/ohio_parser.py#L180).

**Dibaca (10 kanal):**

| Kanal XML | Atribut timestamp | Atribut nilai | Baris kode | Menjadi kolom |
|-----------|-------------------|---------------|-----------|---------------|
| `glucose_level` | `ts` | `value` | [162](../src/data/ohio_parser.py#L162) | `glucose` (timeline dasar) |
| `finger_stick` | `ts` | `value` | [180](../src/data/ohio_parser.py#L180) | `glucose` (timeline SMBG) |
| `meal` | `ts` | `carbs` | [118](../src/data/ohio_parser.py#L118) | `carbs` |
| `bolus` | `ts_begin` | `dose` | [119](../src/data/ohio_parser.py#L119) | `bolus_dose` |
| `basal` | `ts` | `value` | [120](../src/data/ohio_parser.py#L120) | `basal_rate` |
| `exercise` | `ts` | `intensity` | [121](../src/data/ohio_parser.py#L121) | `activity` |
| `stressors` | `ts` | — (presence) | [122](../src/data/ohio_parser.py#L122) | `stress` |
| `sleep` | `ts` | — (presence) | [123](../src/data/ohio_parser.py#L123) | `sleep` |
| `work` | `ts` | — (presence) | [124](../src/data/ohio_parser.py#L124) | `work` |
| `illness` | `ts` | — (presence) | [125](../src/data/ohio_parser.py#L125) | `illness` |

**Diabaikan (9 kanal, ada di XML tetapi tidak pernah dirujuk kode mana pun):**

| Kanal XML | Jumlah event (pasien 559, training) |
|-----------|-------------------------------------|
| `temp_basal` | 34 |
| `hypo_event` | 7 |
| `basis_heart_rate` | 11.979 |
| `basis_gsr` | 11.769 |
| `basis_skin_temperature` | 11.842 |
| `basis_air_temperature` | 11.842 |
| `basis_steps` | 12.288 |
| `basis_sleep` | 1.087 |
| (atribut `type`, `duration`, `competitive` pada `exercise`; `quality` pada `sleep`; `type` pada `meal`; `bwz_carb_input` pada `bolus`) | tidak diekstrak |

Perhatikan bahwa `sleep` dan `work` di XML sebenarnya memakai `ts_begin`/`ts_end`, tetapi [ohio_parser.py:123-124](../src/data/ohio_parser.py#L123-L124) meminta atribut `"ts"`. Karena `_parse_ts()` mengembalikan `None` untuk atribut yang tidak ada ([ohio_parser.py:32-33](../src/data/ohio_parser.py#L32-L33)) dan event tersebut di-`continue` ([ohio_parser.py:49-50](../src/data/ohio_parser.py#L49-L50)), **kolom `sleep` dan `work` selalu bernilai 0**. Diverifikasi pada CSV keluaran: kedua kolom seluruhnya nol.

### A.3 Penyelarasan event ke grid CGM

**Toleransi waktu:** `_BIN_TOL = pd.Timedelta("2min30s")` — [ohio_parser.py:27](../src/data/ohio_parser.py#L27). Yaitu ±2,5 menit, setengah dari cadence CGM 5 menit.

**Mekanisme:** `pandas.merge_asof` dengan `direction="nearest"`, event sebagai tabel kiri dan baris CGM sebagai tabel kanan:

```python
# src/data/ohio_parser.py:70-73
matched = pd.merge_asof(
    events[["ts", value_col]].sort_values("ts"),
    base_ref, on="ts", direction="nearest", tolerance=_BIN_TOL,
).dropna(subset=["_idx"])
```

Event yang tidak menemukan baris CGM dalam ±2,5 menit dibuang oleh `.dropna(subset=["_idx"])` ([ohio_parser.py:73](../src/data/ohio_parser.py#L73)).

**Jika ada lebih dari satu event dalam satu bin:**

- **Nilai kuantitatif (carbs, bolus dose, exercise intensity)** — **dijumlahkan**:
  ```python
  # src/data/ohio_parser.py:76-77
  summed = matched.groupby("_idx")[value_col].sum()
  out.loc[summed.index] = summed.values
  ```
  Konsekuensi: dua bolus dalam 5 menit menjadi satu nilai `bolus_dose` gabungan; dua exercise event menjadi penjumlahan `intensity` (bukan rata-rata, bukan maksimum).

- **Kanal presence (stress, sleep, work, illness)** — **di-dedup menjadi flag 1**:
  ```python
  # src/data/ohio_parser.py:94
  out.loc[matched["_idx"].unique()] = 1
  ```

- **Basal** — tidak memakai binning sama sekali, lihat A.4.

### A.4 Penanganan basal dan temp_basal

**Basal diperlakukan sebagai step function.** Fungsi `_basal_rate_stepwise()` ([ohio_parser.py:98-110](../src/data/ohio_parser.py#L98-L110)) memakai `merge_asof` dengan `direction="backward"` dan **tanpa `tolerance`**:

```python
# src/data/ohio_parser.py:104-109
merged = pd.merge_asof(
    base_ref, basal[["ts", "value"]].sort_values("ts"),
    on="ts", direction="backward",
)
merged["value"] = merged["value"].fillna(0.0)
out.loc[merged["_idx"].values] = merged["value"].values
```

Artinya: rate basal terakhir yang tercatat berlaku terus untuk semua baris CGM sesudahnya, sampai ada event basal berikutnya. Baris CGM sebelum event basal pertama mendapat 0,0.

**Konversi rate → dosis terkirim per langkah:**
```python
# src/data/ohio_parser.py:28
_STEP_HOURS = 5.0 / 60.0             # 1 langkah CGM = 5 menit

# src/data/ohio_parser.py:129-131
basal_rate = _basal_rate_stepwise(basal, ts)
basal_delivered = basal_rate * _STEP_HOURS
insulin = bolus_dose + basal_delivered
```

Kolom `insulin` = bolus + basal terkirim; kolom `basal_rate` disimpan terpisah sebagai rate (unit/jam).

**Apakah `temp_basal` menimpa basal: TIDAK.** `temp_basal` **tidak pernah dibaca sama sekali**. Pencarian string `temp_basal` di seluruh `*.py`, `*.md`, `*.yaml` proyek mengembalikan **nol hasil**. Kanal ini ada di XML (34 event pada pasien 559 saja, dengan atribut `ts_begin`, `ts_end`, `value`) tetapi diabaikan sepenuhnya. Konsekuensinya, periode temp basal (termasuk suspensi pompa, `value="0.0"`) tetap dihitung memakai rate basal terjadwal.

### A.5 Penanganan nilai kosong

Dua tempat berbeda:

**(1) Saat parsing** — [src/data/ohio_parser.py](../src/data/ohio_parser.py):
- Atribut yang hilang / tidak bisa di-`float` → **diisi 0.0**, bukan NaN ([ohio_parser.py:54-57](../src/data/ohio_parser.py#L54-L57)).
- Event dengan timestamp tak terparse → **dibuang** ([ohio_parser.py:49-50](../src/data/ohio_parser.py#L49-L50)).
- `glucose <= 0` → **dibuang** ([ohio_parser.py:165](../src/data/ohio_parser.py#L165) untuk CGM, [ohio_parser.py:183](../src/data/ohio_parser.py#L183) untuk finger_stick).

**(2) Saat pelatihan** — `DataPreprocessor.handle_missing_values(max_interpolate_steps=6)`:

- **Metode:** interpolasi **linear** pada semua kolom numerik dengan
  `limit=max_interpolate_steps`; runtun NaN yang lebih panjang **dibuang barisnya**,
  bukan diisi paksa.
- **Batas: `model.max_interpolate_steps = 6`** (30 menit) dari `config.yaml`.

> **BERUBAH (Tugas 5) — TETAPI PREMISNYA TIDAK BERLAKU.** Versi 1 mencatat
> "gap sepanjang apa pun diinterpolasi". Batas kini ada, **namun efeknya NOL pada
> `ohio_t1dm_merged.csv`**: berkas itu memuat **nol NaN** di seluruh 14 kolom.
> Parser sudah mengisi atribut yang hilang dengan 0.0 dan membuang baris glukosa
> tidak valid, sehingga baris yang "hilang" akibat jeda sensor memang **tidak ada**
> di dalam berkas — bukan hadir sebagai NaN. `interpolate()` tidak pernah
> menyentuhnya. Batas ini berfungsi sebagai pengaman untuk `manual_logbook.csv`.

### A.5b Segmentasi jendela pada jeda sensor

> **BARU (Tugas 5).** Inilah cacat yang sesungguhnya, yang tersembunyi di balik
> premis "interpolasi tanpa batas" pada Versi 1.

`create_sequences()` membentuk jendela per **posisi baris**, bukan per waktu.
Distribusi jeda antar-baris (166.521 interval):

| Jeda | Jumlah | % |
|---|---|---|
| ≤5 mnt (normal) | 165.497 | 99,385% |
| 5–15 mnt | 537 | 0,322% |
| 15–60 mnt | 220 | 0,132% |
| 60–360 mnt | 195 | 0,117% |
| >360 mnt | 72 | 0,043% |

Jeda terpanjang **7.080 menit (118 jam)**. Tanpa penyaringan, jendela memperlakukan
lompatan itu sebagai satu langkah 5 menit — model belajar dari kesinambungan yang
tidak pernah ada.

```python
# Penanda jeda terlalu panjang; jumlah kumulatif membuat pemeriksaan O(1).
dt = patient_df['timestamp'].diff().dt.total_seconds().div(60.0).to_numpy()
bad = np.zeros(len(data), dtype=np.int32)
bad[1:] = (dt[1:] > max_gap_min + 1e-9).astype(np.int32)
bad_cum = np.cumsum(bad)
...
if bad_cum[i + span - 1] - bad_cum[i] > 0:
    n_dibuang += 1
    continue
```

Rentang yang diperiksa mencakup **seluruh baris `i..i+span-1`** — jendela masukan
**dan** jalur menuju target di horizon. Kalau hanya jendela masukan yang diperiksa,
target bisa berada di seberang jeda dan labelnya salah.

**Dampak** (`max_gap_steps = 6`, jeda maksimum 30 menit):

| Horizon | Himpunan | Tanpa | Dengan | Dibuang | % |
|---|---|---|---|---|---|
| 6 | latih | 139.136 | 133.636 | 5.500 | 3,95% |
| 6 | uji | 27.193 | 26.445 | 748 | 2,75% |
| 12 | latih | 139.076 | 131.646 | 7.430 | 5,34% |
| 12 | uji | 27.181 | 26.169 | 1.012 | 3,72% |

**Catatan tambahan yang MASIH BERLAKU:** interpolasi dilakukan pada DataFrame yang
sudah di-sort `['patient_id','timestamp']` tetapi **tidak di-`groupby('patient_id')`**.
Segmentasi jendela sendiri sudah per pasien (lewat `groupby` di `create_sequences`),
tetapi `handle_missing_values()` belum. Pada dataset ini tidak berdampak (nol NaN).

Selain itu, `validate_data_contract()` membuang baris dengan `patient_id` atau `timestamp` NaN ([src/data/contracts.py:77](../src/data/contracts.py#L77)).

### A.6 Jumlah pasien yang benar-benar dipakai

**Dari kode dan data: 12 pasien, semuanya dipakai. Tidak ada pasien yang dibuang.**

Verifikasi langsung `data/raw/ohio_t1dm_merged.csv` — 166.533 baris, 12 pasien:

| patient_id | baris | patient_id | baris |
|---|---|---|---|
| ohio_540 | 14.843 | ohio_570 | 13.727 |
| ohio_544 | 13.339 | ohio_575 | 14.456 |
| ohio_552 | 11.444 | ohio_584 | 14.815 |
| ohio_559 | 13.310 | ohio_588 | 15.431 |
| ohio_563 | 14.694 | ohio_591 | 13.607 |
| ohio_567 | 13.247 | ohio_596 | 13.620 |

24 file XML (12 pasien × training + testing) di-glob oleh [ohio_parser.py:217](../src/data/ohio_parser.py#L217), dan seluruhnya digabung tanpa penyaringan pasien.

**Pembagian pada pelatihan utama — 10 latih / 2 uji:**
```python
# src/models/rf_model.py:136-141
patient_ids = sorted(df["patient_id"].unique().tolist())
...
test_patients = patient_ids[-2:]
train_df, test_df = preprocessor.split_by_patient(df, test_patients)
```
`patient_ids[-2:]` pada daftar terurut = `["ohio_591", "ohio_596"]`. Jadi **10 pasien latih, 2 pasien uji**. Pola `patients[-2:]` yang identik dipakai di [lstm_model.py:194](../src/models/lstm_model.py#L194), [eval_rf_lstm.py:153](../scripts/eval_rf_lstm.py#L153), [train_condition_classifier.py:61](../scripts/train_condition_classifier.py#L61), dan skrip evaluasi lain.

**Asal angka "8 dari 10" — kemungkinan besar tercampur dari skrip conformal.** Satu-satunya tempat di kode yang menghasilkan angka 8 adalah [scripts/conformal_calibration.py:59](../scripts/conformal_calibration.py#L59):

```python
# scripts/conformal_calibration.py:59
test_p, cal_p, train_p = pids[-2:], pids[-4:-2], pids[:-4]
```

Yaitu **8 latih / 2 kalibrasi / 2 uji** — pembagian tiga arah yang hanya berlaku untuk kalibrasi conformal, bukan untuk pelatihan model utama.

**Kesimpulan untuk laporan:** klaim `METHODOLOGY.md:147` ("Train: 80% pasien (8 dari 10 yang digunakan)") **tidak didukung kode**. Yang benar menurut kode:
- Dataset: **12 pasien**, semuanya diparse dan dipakai.
- Pelatihan RF/LSTM utama: **10 latih / 2 uji** (`ohio_591`, `ohio_596` sebagai hold-out).
- Kalibrasi conformal saja: **8 latih / 2 kalibrasi / 2 uji**.
- **Tidak ada satu pasien pun yang dibuang, dan tidak ada kriteria eksklusi pasien di kode mana pun.**

---

## B. REKAYASA FITUR

Seluruhnya di `DataPreprocessor.engineer_features()` — [src/data/preprocessor.py:78-118](../src/data/preprocessor.py#L78-L118), dengan pembantu `_decay_accumulate()` di [preprocessor.py:17-28](../src/data/preprocessor.py#L17-L28).

### B.1 Fungsi IOB dan COB

**Path:** `src/data/preprocessor.py`
**Fungsi inti:** `_decay_accumulate()` (modul-level, baris 17-28) dipanggil dari `engineer_features()` (baris 105-106).

Rumus akumulasi peluruhan (salinan persis):
```python
# src/data/preprocessor.py:17-28
def _decay_accumulate(values: np.ndarray, decay: float) -> np.ndarray:
    """Akumulasi peluruhan eksponensial first-order: out[t] = values[t] + decay*out[t-1]."""
    out = np.zeros(len(values), dtype=float)
    acc = 0.0
    for i, v in enumerate(values):
        acc = float(v) + decay * acc
        out[i] = acc
    return out
```

Konstanta peluruhan (salinan persis):
```python
# src/data/preprocessor.py:98-99
insulin_decay = float(np.exp(-source_interval_min / insulin_tau_min))
carbs_decay = float(np.exp(-source_interval_min / carbs_tau_min))
```

Pemanggilan per pasien (salinan persis):
```python
# src/data/preprocessor.py:102-106
for _, g in df.groupby("patient_id", sort=False):
    g = g.copy()
    insulin_src = g["bolus_dose"] if "bolus_dose" in g.columns else g["insulin"]
    g["iob"] = _decay_accumulate(insulin_src.to_numpy(dtype=float), insulin_decay)
    g["cob"] = _decay_accumulate(g["carbs"].to_numpy(dtype=float), carbs_decay)
```

Ringkasan matematis:

```
decay_insulin = exp(-5 / 240) = 0.979296…
decay_carbs   = exp(-5 / 180) = 0.972604…

IOB[t] = bolus_dose[t] + decay_insulin · IOB[t-1]
COB[t] = carbs[t]      + decay_carbs   · COB[t-1]
IOB[-1] = COB[-1] = 0   (acc diinisialisasi 0.0 per pasien)
```

Dua catatan yang perlu akurat di laporan:
1. **IOB memakai `bolus_dose`, bukan `insulin`** ([preprocessor.py:104](../src/data/preprocessor.py#L104)). Komponen basal sengaja tidak masuk IOB. Fallback ke `insulin` hanya jika kolom `bolus_dose` tidak ada.
2. Ini adalah **akumulator IIR orde-1**, bukan kurva farmakokinetik bi-eksponensial. Nilai IOB tidak berdimensi "unit insulin tersisa" secara fisiologis — ia adalah jumlah tertimbang eksponensial dari dosis lampau.

### B.2 Konstanta waktu insulin dan karbohidrat

**Nilai yang benar-benar dipakai** (verifikasi dari `models/rf_inference_bundle.pkl`, kunci `feature_engineering`):

```
insulin_tau_min : 240   (4 jam)
carbs_tau_min   : 180   (3 jam)
trend_steps     : 3     (15 menit)
```

**Sumber nilai: `config.yaml`**, bukan hardcoded pada jalur produksi.

```yaml
# config.yaml:53-56
  feature_engineering:
    insulin_tau_min: 240   # waktu kerja insulin ~4 jam
    carbs_tau_min: 180     # penyerapan karbohidrat ~3 jam
    trend_steps: 3         # tren glukosa atas 3 langkah (15 menit)
```

Alur pembacaan:
```python
# src/models/rf_model.py:128-130
fe_params = config["model"].get("feature_engineering", {})
if use_engineered:
    df = preprocessor.engineer_features(df, **fe_params)
```

Nilai yang sama **juga** muncul sebagai default hardcoded pada tanda tangan fungsi ([preprocessor.py:81-84](../src/data/preprocessor.py#L81-L84)): `insulin_tau_min=240.0`, `carbs_tau_min=180.0`, `trend_steps=3`, `source_interval_min=5.0`. Default ini yang aktif jika `engineer_features()` dipanggil tanpa argumen.

Satu konstanta **selalu hardcoded**: `source_interval_min = 5.0` ([preprocessor.py:84](../src/data/preprocessor.py#L84)). Nilai ini **tidak ada di `config.yaml`** dan tidak pernah dioper oleh pemanggil mana pun — termasuk saat aplikasi menghitung fitur untuk data SMBG. Berbeda dari `config.yaml:7 sampling_interval_min: 5`, yang tidak dibaca oleh jalur ini.

**Konstanta terpisah di modul Digital Twin** (tidak dipakai model prediksi) — [src/digital_twin/patient_twin.py:52-59](../src/digital_twin/patient_twin.py#L52-L59): `insulin_duration_hours: 4.0`, `carb_absorption_hours: 3.0`, hardcoded di konstruktor.

### B.3 Rumus glucose_delta

**Selisih terhadap titik sebelumnya (lag 3 langkah), BUKAN rata-rata bergerak.**

```python
# src/data/preprocessor.py:107
g["glucose_delta"] = g["glucose"].diff(trend_steps).fillna(0.0)
```

Dengan `trend_steps = 3` dan cadence 5 menit:

```
glucose_delta[t] = glucose[t] - glucose[t-3]        (selisih atas 15 menit)
glucose_delta[t] = 0.0   untuk t < 3                (fillna)
```

Dihitung per pasien di dalam loop `groupby("patient_id")` ([preprocessor.py:102](../src/data/preprocessor.py#L102)), sehingga tidak ada kebocoran antar pasien.

### B.4 Rumus hour_sin dan hour_cos

Salinan persis:
```python
# src/data/preprocessor.py:108-110
hour = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60.0
g["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
g["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
```

```
hour      = jam + menit/60                (nilai kontinu 0–24, detik diabaikan)
hour_sin  = sin(2π · hour / 24)
hour_cos  = cos(2π · hour / 24)
```

### B.5 Pembentukan kolom activity

```python
# src/data/ohio_parser.py:121
exercise = _events_to_df(root, "exercise", "ts", ["intensity"])

# src/data/ohio_parser.py:132
activity = _align_sum(exercise, ts, "intensity")
```

- Sumber tunggal: kanal XML `exercise`, atribut **`intensity`**.
- Atribut `duration` pada event `exercise` **tidak dibaca** — meskipun ada di XML (contoh: `duration='150'`).
- Kanal `work` **tidak** masuk ke `activity`. Ia diproses terpisah sebagai flag presence ke kolom `work` ([ohio_parser.py:145](../src/data/ohio_parser.py#L145)) — dan seperti dijelaskan di A.2, kolom itu selalu 0 karena salah nama atribut.
- Jika beberapa event exercise jatuh di bin yang sama, nilai `intensity` **dijumlahkan** ([ohio_parser.py:76-77](../src/data/ohio_parser.py#L76-L77)).

**Konsekuensi untuk penulisan laporan:** kolom `activity` berisi **skor intensitas** (skala ordinal OhioT1DM, umumnya 1–10), **bukan menit aktivitas**. Namun UI dan `PatientState` memperlakukannya sebagai menit: `activity_level: int = 0  # minutes today` ([src/patient_state.py:68](../src/patient_state.py#L68)), dan aplikasi menampilkannya sebagai `"Aktivitas: {…} mnt"` ([app/streamlit_app.py:203](../app/streamlit_app.py#L203)). Ini ketidakcocokan satuan yang nyata di kode.

### B.6 Urutan kolom fitur akhir yang masuk ke model

Dari `config.yaml:51`, dan diverifikasi dari kunci `features` di `models/rf_inference_bundle.pkl`:

```
index 0 : glucose
index 1 : glucose_delta
index 2 : iob
index 3 : cob
index 4 : activity
index 5 : hour_sin
index 6 : hour_cos
```

7 fitur. Urutan ini bersifat mengikat: `create_sequences()` memakai **index 0 sebagai anchor glukosa** ([preprocessor.py:161-163](../src/data/preprocessor.py#L161-L163)), sehingga `glucose` **harus** berada di posisi pertama.

Penetapan urutan:
```python
# src/models/rf_model.py:131-134
    feature_list = config["model"]["engineered_features"]
else:
    feature_list = config["model"]["features"]
preprocessor.feature_columns = list(feature_list)
```

Konfigurasi non-engineered (`config.yaml:45`, tidak aktif karena `use_engineered: true`): `["glucose", "carbs", "insulin", "activity"]`.

Bentuk tensor masuk model: `(n_samples, 12, 7)` → di-flatten menjadi `(n_samples, 84)` untuk Random Forest (lihat C.4).

### B.7 Target: Δglukosa dan rekonstruksi

**Konfirmasi: ya, target adalah Δ relatif nilai glukosa terakhir dalam window.**

Anchor dan target mentah dibentuk di `create_sequences()`:
```python
# src/data/preprocessor.py:158-163
for i in range(len(data) - span + 1):
    X.append(data[i:i + sequence_length])
    # target = glukosa pada (akhir window + horizon)
    y.append(data[i + sequence_length + prediction_horizon - 1, 0])
    # anchor = glukosa terakhir di window (untuk target delta), fitur index 0 = glucose
    anchors.append(data[i + sequence_length - 1, 0])
```

Transformasi ke delta saat pelatihan:
```python
# src/models/rf_model.py:149
y_train_fit = (y_train - anc_train) if predict_delta else y_train
```

Rekonstruksi saat evaluasi:
```python
# src/models/rf_model.py:153-155
y_pred = model.predict(X_test_scaled)
if predict_delta:
    y_pred = y_pred + anc_test
```

Rekonstruksi saat **inferensi di aplikasi** (salinan persis):
```python
# app/streamlit_app.py:101-109
def predict_next(window_df, art):
    """Prediksi glukosa absolut; rekonstruksi dari delta bila model dilatih delta."""
    X = window_df[art["features"]].values.astype(float)
    if art["scaler"] is not None:
        X = art["scaler"].transform(X)
    out = float(art["model"].predict(X.reshape(1, -1))[0])
    if art["predict_delta"]:
        out += float(window_df["glucose"].iloc[-1])  # anchor = glukosa terakhir window
    return out
```

Ringkasan:
```
anchor        = glucose[akhir window]
y_latih       = glucose[akhir window + horizon] - anchor
prediksi_abs  = model.predict(X) + anchor
```

Flag `predict_delta` dibaca dari `config.yaml:52` (`predict_delta: true`) dan **disimpan di dalam bundle** ([rf_model.py:171](../src/models/rf_model.py#L171)), sehingga aplikasi tahu harus merekonstruksi tanpa membaca config.

---

## C. MODEL PREDIKSI

### C.1 Hyperparameter Random Forest

Kelas: `RandomForestGlucoseModel` — [src/models/rf_model.py:21-87](../src/models/rf_model.py#L21-L87).

```python
# src/models/rf_model.py:28-34
self.model = RandomForestRegressor(
    n_estimators=rf_cfg.get("n_estimators", 200),
    max_depth=rf_cfg.get("max_depth", 20),
    min_samples_split=rf_cfg.get("min_samples_split", 5),
    random_state=(config or {}).get("data", {}).get("seed", 42),
    n_jobs=-1,
)
```

| Parameter | Nilai efektif | Sumber |
|-----------|---------------|--------|
| `n_estimators` | **200** | `config.yaml:63` |
| `max_depth` | **20** | `config.yaml:64` |
| `min_samples_split` | **5** | `config.yaml:65` |
| `random_state` | **42** | `config.yaml:9` (`data.seed`) — perhatikan: dibaca dari blok `data`, bukan `model.random_forest` |
| `n_jobs` | **-1** | hardcoded, [rf_model.py:33](../src/models/rf_model.py#L33) |

Parameter lain (`min_samples_leaf`, `max_features`, `bootstrap`, `criterion`) **TIDAK DITEMUKAN** — memakai default scikit-learn.

Metrik hold-out (`models/rf_metrics_h{6,12}.json`):

> **BERUBAH (Tugas 5):** angka diukur ulang setelah jendela yang melintasi jeda
> sensor dibuang. Perbaikan konsisten di kedua horizon.

| Horizon | Metrik | Versi 1 | **Versi 2** | Δ |
|---|---|---|---|---|
| **+30 mnt** | RMSE | 22,598 | **21,123** | −1,475 |
| | MAE | 15,098 | — | |
| | Clarke A+B | 94,351% | **94,857%** | +0,51 pp |
| **+60 mnt** | RMSE | 34,240 | **32,729** | −1,511 |
| | MAE | 24,825 | — | |
| | Clarke A+B | 86,939% | **87,822%** | +0,88 pp |

**Durasi pelatihan** (Ryzen 5 5600H, 6C/12T, tanpa GPU): horizon 6 **968 detik**,
horizon 12 **1.203 detik**, pengklasifikasi kondisi **122 detik** — total **38 menit
13 detik**.

**PERANCU yang harus disebut di laporan.** Perbandingan ini **bukan apel-ke-apel**:
himpunan uji ikut menyusut (748 jendela pada h6, 1.012 pada h12). Sebagian perbaikan
berasal dari **hilangnya kasus uji yang tidak sah**, bukan dari model yang benar-benar
lebih baik, dan kedua efek itu tidak dapat dipisahkan dari data yang ada.

Yang dapat dinyatakan tanpa syarat: angka Versi 2 diukur pada himpunan uji yang
**seluruh jendelanya sah secara temporal**, sedangkan angka Versi 1 sebagian ditopang
jendela yang memperlakukan lompatan hingga 118 jam sebagai satu langkah 5 menit.

### C.2 Arsitektur LSTM pembanding

Kelas: `LSTMGlucoseModel` — [src/models/lstm_model.py:24-135](../src/models/lstm_model.py#L24-L135).

Arsitektur (salinan persis):
```python
# src/models/lstm_model.py:44-55
inp = keras.Input(shape=input_shape, name="glucose_window")
x = keras.layers.LSTM(self.units_1, return_sequences=True, name="lstm_1")(inp)
x = keras.layers.Dropout(self.dropout, name="drop_1")(x)
x = keras.layers.LSTM(self.units_2, return_sequences=False, name="lstm_2")(x)
x = keras.layers.Dropout(self.dropout, name="drop_2")(x)
out = keras.layers.Dense(1, name="glucose_pred")(x)
model = keras.Model(inp, out)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
    loss="mse",
    metrics=["mae"],
)
```

| Aspek | Nilai | Sumber |
|-------|-------|--------|
| Jumlah layer LSTM | **2** (stacked, tanpa attention) | [lstm_model.py:45,47](../src/models/lstm_model.py#L45-L47) |
| Unit layer 1 | **64** (`return_sequences=True`) | `config.yaml:68` |
| Unit layer 2 | **32** (`return_sequences=False`) | `config.yaml:69` |
| Dropout | **0.2**, dua kali (setelah tiap LSTM) | `config.yaml:70` |
| Output | `Dense(1)`, linear | [lstm_model.py:49](../src/models/lstm_model.py#L49) |
| Optimizer | **Adam**, `learning_rate=0.001` | `config.yaml:71` |
| Loss | **MSE**; metrik pantau **MAE** | [lstm_model.py:53-54](../src/models/lstm_model.py#L53-L54) |
| Epoch maksimum | **100** | `config.yaml:72` |
| EarlyStopping patience | **10**, `restore_best_weights=True` | `config.yaml:73`, [lstm_model.py:78-83](../src/models/lstm_model.py#L78-L83) |
| Batch size | **adaptif**: `max(8, min(32, len(X_train)//10))` | hardcoded [lstm_model.py:75](../src/models/lstm_model.py#L75) |

Dua perbedaan penting antara jalur LSTM dan jalur RF yang harus disebutkan agar perbandingan di laporan tidak menyesatkan:

1. **LSTM tidak memakai fitur engineered dan tidak memakai target delta.** `train_lstm_from_config()` tidak pernah memanggil `engineer_features()`, memanggil `create_sequences()` tanpa `return_anchor` ([lstm_model.py:197-198](../src/models/lstm_model.py#L197-L198)), dan menyimpan `"features": config["model"]["features"]` ([lstm_model.py:216](../src/models/lstm_model.py#L216)) — yaitu 4 fitur mentah `["glucose","carbs","insulin","activity"]`, target absolut.
2. **`prediction_horizon` tidak dioper.** [lstm_model.py:197-198](../src/models/lstm_model.py#L197-L198) memakai default `prediction_horizon=1` (+5 menit), sedangkan RF dilatih pada horizon 6 dan 12.

Perbandingan RF-vs-LSTM yang setara ada di [scripts/eval_rf_lstm.py](../scripts/eval_rf_lstm.py) dan [scripts/crossval_rf_vs_lstm.py](../scripts/crossval_rf_vs_lstm.py), bukan pada `train_lstm_from_config()`.

### C.3 Satu model untuk semua pasien, atau per pasien

**Satu model populasi untuk semua pasien.** Tidak ada pelatihan per pasien di mana pun.

Buktinya: `create_sequences()` melakukan `groupby('patient_id')` hanya untuk **mencegah window melintasi batas pasien** ([preprocessor.py:151-163](../src/data/preprocessor.py#L151-L163)), lalu seluruh window dari semua pasien digabung menjadi satu array `X` dan satu panggilan `model.train(X_train_scaled, y_train_fit)` ([rf_model.py:152](../src/models/rf_model.py#L152)).

Tidak ada identitas pasien yang masuk sebagai fitur — `patient_id` tidak ada dalam `engineered_features`. Model bersifat **patient-agnostic**, dan pemisahan uji dilakukan **per pasien** ([preprocessor.py:262-286](../src/data/preprocessor.py#L262-L286)) sehingga 2 pasien uji benar-benar tak pernah dilihat saat latih.

Satu file model disimpan per horizon, bukan per pasien: `rf_baseline_h6.pkl`, `rf_baseline_h12.pkl`.

### C.4 Penerapan sequence_length=12 pada Random Forest

**Ya, flatten.** Method `_flatten()` — [src/models/rf_model.py:36-40](../src/models/rf_model.py#L36-L40):

```python
def _flatten(self, X: np.ndarray) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError("Expected 3D input (samples, sequence_length, n_features)")
    n_samples, seq_len, n_features = X.shape
    return X.reshape(n_samples, seq_len * n_features)
```

Dipanggil di `train()` ([rf_model.py:53](../src/models/rf_model.py#L53)) dan `predict()` ([rf_model.py:73](../src/models/rf_model.py#L73)).

```
(n_samples, 12, 7)  →  reshape  →  (n_samples, 84)
```

Urutan kolom hasil flatten adalah row-major (C-order): `[t0_glucose, t0_glucose_delta, …, t0_hour_cos, t1_glucose, …, t11_hour_cos]`. Yaitu **12 blok waktu, tiap blok 7 fitur**.

**Normalisasi dilakukan sebelum flatten**, pada bentuk 3D, dengan `StandardScaler` yang di-fit atas seluruh titik waktu tergabung ([preprocessor.py:203-211](../src/data/preprocessor.py#L203-L211)): scaler di-fit pada `X_train.reshape(-1, n_features)`, jadi **satu mean/std per fitur, dibagi bersama seluruh 12 langkah waktu** — bukan 84 parameter terpisah.

Aplikasi melakukan flatten yang setara dengan `X.reshape(1, -1)` ([app/streamlit_app.py:106](../app/streamlit_app.py#L106)).

### C.5 Isi bundle inferensi

**Format: pickle** (`pickle.dump`, protokol default).
**Path:** `models/rf_inference_bundle_h{horizon}.pkl`, ditambah salinan kanonik `models/rf_inference_bundle.pkl` untuk horizon default.

Kode penyimpanan (salinan persis):
```python
# src/models/rf_model.py:164-183
bundle = {
    "model": model.model,
    "scaler": preprocessor.scaler,
    "features": feature_list,
    "sequence_length": sequence_length,
    "prediction_horizon": horizon,
    "use_engineered": use_engineered,
    "predict_delta": predict_delta,
    "feature_engineering": fe_params,
}
with open(models_dir / f"rf_inference_bundle_h{horizon}.pkl", "wb") as f:
    pickle.dump(bundle, f)
with open(models_dir / f"rf_metrics_h{horizon}.json", "w", encoding="utf-8") as f:
    json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

# Horizon default → simpan juga dengan nama kanonik untuk aplikasi
if horizon == default_horizon:
    model.save(str(models_dir / "rf_baseline.pkl"))
    with open(models_dir / "rf_inference_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)
```

Isi aktual `models/rf_inference_bundle.pkl` (dibaca langsung):

| Kunci | Tipe | Nilai |
|-------|------|-------|
| `model` | `sklearn.ensemble.RandomForestRegressor` | 200 pohon terlatih |
| `scaler` | `sklearn.preprocessing.StandardScaler` | di-fit atas 7 fitur |
| `features` | `list[str]` | `['glucose','glucose_delta','iob','cob','activity','hour_sin','hour_cos']` |
| `sequence_length` | `int` | `12` |
| `prediction_horizon` | `int` | `6` (= +30 menit) |
| `use_engineered` | `bool` | `True` |
| `predict_delta` | `bool` | `True` |
| `feature_engineering` | `dict` | `{'insulin_tau_min': 240, 'carbs_tau_min': 180, 'trend_steps': 3}` |

**Yang TIDAK disimpan dalam bundle:** metrik, versi scikit-learn, daftar pasien latih/uji, timestamp pelatihan, dan koefisien conformal. Metrik ditulis ke file JSON terpisah.

Berkas artefak lain di `models/`:
- `rf_baseline.pkl`, `rf_baseline_h6.pkl`, `rf_baseline_h12.pkl` — model **telanjang** (hanya `RandomForestRegressor`, tanpa scaler/metadata), disimpan `RandomForestGlucoseModel.save()` ([rf_model.py:76-81](../src/models/rf_model.py#L76-L81)).
- `rf_inference_bundle_h12.pkl` — bundle horizon +60 menit.
- `rf_condition_classifier_h6.pkl` — bundle pengklasifikasi kondisi, kunci: `model`, `scaler`, `classes`, `features`, `sequence_length`, `prediction_horizon` ([train_condition_classifier.py:156-159](../scripts/train_condition_classifier.py#L156-L159)).
- Bundle LSTM: direktori `models/lstm_bundle/` berisi `keras_model/` + `meta.pkl` dengan kunci `scaler`, `features`, `sequence_length`, `mode` ([lstm_model.py:214-221](../src/models/lstm_model.py#L214-L221)). **Belum ada di direktori `models/` saat ini** — hanya artefak RF yang tersimpan.

---

## D. KALIBRASI KETIDAKPASTIAN

### D.1 Path dan nama fungsi

**Path:** [scripts/conformal_calibration.py](../scripts/conformal_calibration.py) — satu-satunya implementasi conformal prediction di proyek.

| Fungsi | Lokasi | Peran |
|--------|--------|-------|
| `conformal_q(scores, alpha)` | [conformal_calibration.py:35-38](../scripts/conformal_calibration.py#L35-L38) | Kuantil konformal dengan koreksi finite-sample |
| `rf_std(model, Xf)` | [conformal_calibration.py:31-32](../scripts/conformal_calibration.py#L31-L32) | Simpangan baku antar-pohon |
| `main()` | [conformal_calibration.py:41-115](../scripts/conformal_calibration.py#L41-L115) | Orkestrasi split, fit, evaluasi cakupan |

**Tidak ada kelas atau modul conformal di dalam `src/`.** Ini skrip evaluasi offline; keluarannya `results/eval_prediksi/conformal.json`.

### D.2 Asal data kalibrasi

**Split terpisah per pasien, tiga arah** (salinan persis):
```python
# scripts/conformal_calibration.py:58-60
pids = sorted(df["patient_id"].unique().tolist())
test_p, cal_p, train_p = pids[-2:], pids[-4:-2], pids[:-4]
print(f"train={len(train_p)} kalibrasi={cal_p} test={test_p}")
```

Dengan 12 pasien terurut:

| Split | Pasien | Jumlah | Proporsi pasien |
|-------|--------|--------|-----------------|
| Latih | `ohio_540` … `ohio_584` | **8** | 66,7% |
| Kalibrasi | `ohio_584`, `ohio_588` → tepatnya `pids[-4:-2]` = `ohio_584`, `ohio_588` | **2** | 16,7% |
| Uji | `ohio_591`, `ohio_596` | **2** | 16,7% |

Jadi proporsi kalibrasi = **2 dari 12 pasien (16,7%)**, dipisahkan **per pasien**, bukan per baris. Ini menirukan deployment untuk pasien baru — pasien kalibrasi tidak pernah dilihat saat latih.

Residual kalibrasi:
```python
# scripts/conformal_calibration.py:82
res_ca = np.abs(yca - yca_p)  # residual kalibrasi
```

Model yang dikalibrasi dilatih ulang di dalam skrip ini ([conformal_calibration.py:72-74](../scripts/conformal_calibration.py#L72-L74)) dengan hyperparameter sama dari `config.yaml`, **bukan** memuat `rf_inference_bundle.pkl`.

### D.3 Rumus penetapan lebar interval dan tingkat kepercayaan

Kuantil konformal dengan koreksi finite-sample (salinan persis):
```python
# scripts/conformal_calibration.py:35-38
def conformal_q(scores, alpha):
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)  # koreksi finite-sample
    return float(np.quantile(scores, level, method="higher"))
```

```
level = min(1, ⌈(n+1)(1-α)⌉ / n)
q     = Quantile_level({ |y_i - ŷ_i| : i ∈ kalibrasi }, metode "higher")
```

Tiga varian interval yang dihitung dan dibandingkan (salinan persis):
```python
# scripts/conformal_calibration.py:91-100
for alpha, tgt in [(0.10, 90), (0.05, 95)]:
    # baseline: +/- z*std
    z = 1.645 if tgt == 90 else 1.96
    lo_b, hi_b = yte_p - z * std_te, yte_p + z * std_te
    # conformal absolut
    q_abs = conformal_q(res_ca, alpha)
    lo_a, hi_a = yte_p - q_abs, yte_p + q_abs
    # conformal ternormalisasi
    q_norm = conformal_q(res_ca / (std_ca + EPS), alpha)
    lo_n, hi_n = yte_p - q_norm * std_te, yte_p + q_norm * std_te
```

| Varian | Rumus | Sifat lebar |
|--------|-------|-------------|
| Baseline (bukan conformal) | `ŷ ± z·σ_pohon`, z = 1,645 / 1,96 | adaptif |
| Conformal absolut | `ŷ ± q_abs` | **seragam** untuk semua sampel |
| Conformal ternormalisasi | `ŷ ± q_norm · σ_pohon`, dengan `q_norm` = kuantil dari `|residual| / σ_kalibrasi` | **adaptif** |

`σ_pohon` = simpangan baku prediksi antar 200 pohon ([conformal_calibration.py:31-32](../scripts/conformal_calibration.py#L31-L32)); `EPS = 1e-6` ([conformal_calibration.py:28](../scripts/conformal_calibration.py#L28)).

**Tingkat kepercayaan yang dipakai: dua level, 90% (α=0,10) dan 95% (α=0,05)** ([conformal_calibration.py:91](../scripts/conformal_calibration.py#L91)).

**Yang dipakai di aplikasi berbeda.** Aplikasi **tidak** memuat `conformal.json`; ia memakai satu faktor pengali yang di-hardcode (salinan persis):
```python
# app/streamlit_app.py:190-195
if pred_std:
    # Faktor conformal ternormalisasi (kalibrasi split-conformal → cakupan ~95.5% tervalidasi).
    # ±1.96·std hanya menutup ~86% (falsely confident); lihat scripts/conformal_calibration.py.
    CONFORMAL_K = 3.3
    lo, hi = pred - CONFORMAL_K * pred_std, pred + CONFORMAL_K * pred_std
    st.caption(f"Rentang keyakinan 95% (terkalibrasi conformal): **{lo:.0f}–{hi:.0f}** mg/dL")
```

Nilai `3.3` juga diulang secara terpisah di [app/streamlit_app.py:221](../app/streamlit_app.py#L221) untuk peringatan hipoglikemia dini. Jadi: **produksi memakai varian conformal ternormalisasi dengan `q_norm ≈ 3,3` yang disalin manual sebagai konstanta**, bukan dibaca dari berkas kalibrasi.

### D.4 Simetris atau tidak

**Simetris, pada ketiga varian dan pada aplikasi.** Semua interval berbentuk `ŷ ± lebar`:
- baseline `yte_p ∓ z*std_te` ([conformal_calibration.py:94](../scripts/conformal_calibration.py#L94))
- absolut `yte_p ∓ q_abs` ([conformal_calibration.py:97](../scripts/conformal_calibration.py#L97))
- ternormalisasi `yte_p ∓ q_norm*std_te` ([conformal_calibration.py:100](../scripts/conformal_calibration.py#L100))
- aplikasi `pred ∓ 3.3*pred_std` ([streamlit_app.py:194](../app/streamlit_app.py#L194))

Skor konformitas memakai **nilai mutlak residual** (`np.abs`, [conformal_calibration.py:82](../scripts/conformal_calibration.py#L82)), bukan residual bertanda. Tidak ada implementasi CQR (Conformalized Quantile Regression) atau kuantil dua sisi terpisah. **Interval asimetris: TIDAK DITEMUKAN.**

---

## E. KLASIFIKASI KONDISI KLINIS

### E.1 Path dan nama kelas

**Path:** [src/patient_state.py](../src/patient_state.py)
**Kelas:** `PatientState` — dataclass, [patient_state.py:34-291](../src/patient_state.py#L34-L291).

Seluruh derivasi kondisi klinis terjadi di `__post_init__()` ([patient_state.py:96-171](../src/patient_state.py#L96-L171)), yang berjalan otomatis saat objek dikonstruksi.

Tiga factory method:

| Method | Lokasi | Sumber masukan |
|--------|--------|----------------|
| `PatientState.from_model_output()` | [patient_state.py:215-255](../src/patient_state.py#L215-L255) | Keluaran RF/LSTM + baris fitur terakhir. **Jalur utama.** |
| `PatientState.from_digital_twin()` | [patient_state.py:257-275](../src/patient_state.py#L257-L275) | Instance `PatientDigitalTwin` + prediksi |
| `PatientState.from_dict()` | [patient_state.py:277-290](../src/patient_state.py#L277-L290) | Deserialisasi dict |

Dipanggil dari [src/rag/pipeline.py:256-266](../src/rag/pipeline.py#L256-L266) dan [src/rag/conditioned_query.py:308-314](../src/rag/conditioned_query.py#L308-L314).

### E.2 Daftar lengkap medan objek

**Medan masukan** (dideklarasikan, [patient_state.py:55-84](../src/patient_state.py#L55-L84)):

| Medan | Tipe | Default | Nilai yang mungkin / rentang |
|-------|------|---------|------------------------------|
| `patient_id` | `str` | — (wajib) | bebas |
| `current_glucose` | `float` | — (wajib) | di-clip ke **[20, 600]** mg/dL ([:98](../src/patient_state.py#L98)) |
| `predicted_glucose` | `float` | — (wajib) | di-clip ke **[20, 600]** mg/dL ([:99](../src/patient_state.py#L99)) |
| `prediction_horizon_minutes` | `int` | `60` | bebas |
| `insulin_on_board` | `float` | `0.0` | di-clamp ≥ 0 ([:100](../src/patient_state.py#L100)) |
| `carbs_on_board` | `float` | `0.0` | di-clamp ≥ 0 ([:101](../src/patient_state.py#L101)) |
| `activity_level` | `int` | `0` | di-clamp ≥ 0 ([:102](../src/patient_state.py#L102)) |
| `stress_level` | `int` | `5` | di-clip ke **[1, 10]** ([:103](../src/patient_state.py#L103)) |
| `predicted_condition` | `Optional[str]` | `None` | `"hypoglycemia"` \| `"normal"` \| `"hyperglycemia"` \| `None` |
| `predicted_lower` | `Optional[float]` | `None` | mg/dL, batas bawah interval konformal |
| `predicted_upper` | `Optional[float]` | `None` | mg/dL, batas atas interval konformal |
| `timestamp` | `str` | `datetime.now().isoformat()` | ISO-8601 |

**Medan turunan** (`field(init=False)`, dihitung di `__post_init__`, [patient_state.py:87-94](../src/patient_state.py#L87-L94)):

| Medan | Tipe | Nilai yang mungkin |
|-------|------|--------------------|
| `glucose_delta` | `float` | `round(predicted - current, 2)` |
| `trend_direction` | `str` | `"rising"` \| `"stable"` \| `"falling"` |
| `trend_label` | `str` | `"meningkat"` \| `"stabil"` \| `"menurun"` |
| `trend_rate` | `str` | `"rapid"` \| `"moderate"` \| `"slow"` |
| `risk_level` | `str` | `"critical_hypoglycemia"` \| `"hypoglycemia"` \| `"normal"` \| `"hyperglycemia"` \| `"critical_hyperglycemia"` |
| `risk_label` | `str` | `"BAHAYA - Hipoglikemia Berat"` \| `"BAHAYA - Hipoglikemia"` \| `"AMAN"` \| `"HATI-HATI - Hiperglikemia"` \| `"BAHAYA - Hiperglikemia Berat"` |
| `urgency` | `str` | `"critical"` \| `"high"` \| `"medium"` \| `"low"` |
| `anticipated_conditions` | `List[str]` | subset terurut dari `["hypoglycemia", "hyperglycemia", "normal"]` |

**Catatan:** `to_dict()` ([patient_state.py:177-196](../src/patient_state.py#L177-L196)) mengekspor 15 medan tetapi **tidak menyertakan** `anticipated_conditions`, `predicted_condition`, `predicted_lower`, `predicted_upper`. `to_rag_context()` ([patient_state.py:198-209](../src/patient_state.py#L198-L209)) hanya mengekspor 5 medan: `current_glucose`, `insulin_on_board`, `carbs_on_board`, `activity_level`, `stress_level`.

### E.3 Ambang persis tiap tingkat risiko

Konstanta ([patient_state.py:21-24](../src/patient_state.py#L21-L24)):
```python
GLUCOSE_LOW = 70.0    # mg/dL — hypoglycemia threshold (ADA)
GLUCOSE_HIGH = 180.0  # mg/dL — hyperglycemia threshold (ADA postprandial)
GLUCOSE_CRITICAL_LOW = 54.0
GLUCOSE_CRITICAL_HIGH = 250.0
```

Logika (salinan persis, [patient_state.py:126-140](../src/patient_state.py#L126-L140)):
```python
if self.predicted_glucose < GLUCOSE_CRITICAL_LOW:
    self.risk_level = "critical_hypoglycemia"
    self.risk_label = "BAHAYA - Hipoglikemia Berat"
elif self.predicted_glucose < GLUCOSE_LOW:
    self.risk_level = "hypoglycemia"
    self.risk_label = "BAHAYA - Hipoglikemia"
elif self.predicted_glucose > GLUCOSE_CRITICAL_HIGH:
    self.risk_level = "critical_hyperglycemia"
    self.risk_label = "BAHAYA - Hiperglikemia Berat"
elif self.predicted_glucose > GLUCOSE_HIGH:
    self.risk_level = "hyperglycemia"
    self.risk_label = "HATI-HATI - Hiperglikemia"
else:
    self.risk_level = "normal"
    self.risk_label = "AMAN"
```

| Rentang `predicted_glucose` (mg/dL) | `risk_level` | `risk_label` |
|-------------------------------------|--------------|--------------|
| `< 54` | `critical_hypoglycemia` | BAHAYA - Hipoglikemia Berat |
| `54 ≤ g < 70` | `hypoglycemia` | BAHAYA - Hipoglikemia |
| `70 ≤ g ≤ 180` | `normal` | AMAN |
| `180 < g ≤ 250` | `hyperglycemia` | HATI-HATI - Hiperglikemia |
| `> 250` | `critical_hyperglycemia` | BAHAYA - Hiperglikemia Berat |

**Penting:** ambang diterapkan pada **`predicted_glucose`, bukan `current_glucose`** ([patient_state.py:124-125](../src/patient_state.py#L124-L125), komentar "this is the novelty").

**Ambang berbeda di tempat lain** yang perlu diketahui agar laporan konsisten:
- [app/ui.py:14-15](../app/ui.py#L14-L15): `GLUCOSE_LOW = 70`, `GLUCOSE_HIGH = 180` — **hanya 3 zona**, tanpa tingkat kritis ([ui.py:24-30](../app/ui.py#L24-L30)).
- [src/rag/pipeline.py:29-34](../src/rag/pipeline.py#L29-L34) `_risk_level_from_prediction()` — juga hanya 3 zona (70/180).
- [scripts/train_condition_classifier.py:47-48](../scripts/train_condition_classifier.py#L47-L48) `classify_glucose()` — 3 kelas (70/180).
- [src/digital_twin/patient_twin.py:264-272](../src/digital_twin/patient_twin.py#L264-L272) — 3 zona (70/180).

Jadi ambang **54 dan 250 hanya hidup di `PatientState`**; seluruh lapisan lain memakai 70/180.

### E.4 Penetapan arah tren

Konstanta ([patient_state.py:26-27](../src/patient_state.py#L26-L27)):
```python
TREND_STABLE_THRESHOLD_MGDL = 10.0  # delta < 10 mg/dL → stable
TREND_RAPID_THRESHOLD_MGDL = 30.0   # delta > 30 mg/dL → rapid change
```

Salinan persis ([patient_state.py:106-122](../src/patient_state.py#L106-L122)):
```python
self.glucose_delta = round(self.predicted_glucose - self.current_glucose, 2)
if abs(self.glucose_delta) < TREND_STABLE_THRESHOLD_MGDL:
    self.trend_direction = "stable"
    self.trend_label = "stabil"
elif self.glucose_delta > 0:
    self.trend_direction = "rising"
    self.trend_label = "meningkat"
else:
    self.trend_direction = "falling"
    self.trend_label = "menurun"

if abs(self.glucose_delta) >= TREND_RAPID_THRESHOLD_MGDL:
    self.trend_rate = "rapid"
elif abs(self.glucose_delta) >= TREND_STABLE_THRESHOLD_MGDL:
    self.trend_rate = "moderate"
else:
    self.trend_rate = "slow"
```

| `|glucose_delta|` (mg/dL) | `trend_rate` | Arah |
|---------------------------|--------------|------|
| `< 10` | `slow` | `stable` / "stabil" |
| `10 ≤ Δ < 30` | `moderate` | `rising` jika Δ>0, `falling` jika Δ<0 |
| `≥ 30` | `rapid` | `rising` / `falling` |

**Ambang perubahan: 10 mg/dL** (stabil vs bergerak) dan **30 mg/dL** (moderat vs cepat). Δ dihitung atas **seluruh horizon prediksi** (prediksi − sekarang), bukan per satuan waktu — jadi ambang yang sama berlaku baik untuk horizon 30 maupun 60 menit.

Ambang tren yang berbeda dipakai di UI: [app/streamlit_app.py:196](../app/streamlit_app.py#L196) memakai `delta > 10` / `delta < -10` tanpa tingkat "rapid".

### E.5 Penetapan tingkat urgensi

Salinan persis ([patient_state.py:163-171](../src/patient_state.py#L163-L171)):
```python
# Urgency
if self.risk_level.startswith("critical"):
    self.urgency = "critical"
elif self.risk_level in ("hypoglycemia", "hyperglycemia") or self.trend_rate == "rapid":
    self.urgency = "high"
elif self.stress_level >= 8 or (self.trend_rate == "moderate" and self.risk_level != "normal"):
    self.urgency = "medium"
else:
    self.urgency = "low"
```

Dievaluasi berurutan, hasil pertama yang cocok menang:

| Urgensi | Kondisi |
|---------|---------|
| `critical` | `risk_level` diawali `"critical"` (yaitu <54 atau >250 mg/dL) |
| `high` | `risk_level` ∈ {hypoglycemia, hyperglycemia} **ATAU** `trend_rate == "rapid"` |
| `medium` | `stress_level ≥ 8` **ATAU** (`trend_rate == "moderate"` **DAN** `risk_level != "normal"`) |
| `low` | selain di atas |

Perhatikan bahwa cabang `medium` yang kedua tidak pernah tercapai: jika `risk_level != "normal"` maka cabang `high` sudah menangkapnya lebih dulu. Jadi secara efektif **`medium` hanya dicapai lewat `stress_level >= 8`**.

Penting juga: urgensi dihitung **setelah** `predicted_condition` menimpa `risk_level` (E.7), sehingga keluaran pengklasifikasi ikut memengaruhi urgensi.

### E.6 Deteksi peringatan divergensi

Ada **dua blok peringatan terpisah** di aplikasi, keduanya di [app/streamlit_app.py](../app/streamlit_app.py). Tidak ada fungsi bernama "divergence" — logikanya inline.

**(a) Peringatan divergen** — salinan persis ([streamlit_app.py:206-209](../app/streamlit_app.py#L206-L209)):
```python
# Peringatan divergen (current normal tapi prediksi bahaya) — nilai jual sistem
if cur_label == "Dalam Target" and pred_label != "Dalam Target":
    st.warning(f"⚠️ **Antisipasi:** kondisi saat ini normal, namun glukosa diprediksi menuju "
               f"**{pred_label}** ({pred:.0f} mg/dL) dalam {horizon_min} menit. Pertimbangkan tindakan pencegahan.")
```

`cur_label` dan `pred_label` berasal dari `classify_glucose()` di [app/ui.py:24-30](../app/ui.py#L24-L30) (ambang 70/180), dipanggil di [streamlit_app.py:171-172](../app/streamlit_app.py#L171-L172):
```python
_, cur_label, _ = classify_glucose(current)
_, pred_label, _ = classify_glucose(pred)
```

Jadi kondisinya: **glukosa sekarang di dalam 70–180, tetapi prediksi keluar dari 70–180**. Arahnya tidak dibedakan (hipo atau hiper sama-sama memicu).

**(b) Peringatan hipoglikemia dini** — salinan persis ([streamlit_app.py:211-229](../app/streamlit_app.py#L211-L229)):
```python
# Peringatan HIPOGLIKEMIA DINI.
# Prediksi titik regresi menyusut ke tengah: pada ambang <70 ia hanya menangkap 14% kejadian
# hipoglikemia. Dua sinyal tambahan dipakai (lihat scripts/train_condition_classifier.py dan
# scripts/eval_retrieval_realcases.py):
#   (a) pengklasifikasi kondisi sadar-biaya  -> sensitivitas hipoglikemia 44% pada ambang standar
#   (b) batas bawah interval konformal       -> menandai risiko yang masih tercakup ketidakpastian
cond_clf = load_condition_classifier()
pred_condition = predict_condition(window_df, cond_clf, art)
lo95 = hi95 = None
if pred_std:
    lo95, hi95 = pred - 3.3 * pred_std, pred + 3.3 * pred_std

if pred_condition == "hypoglycemia" and pred >= 70.0:
    st.warning(f"🔻 **Waspada hipoglikemia:** prediksi titik **{pred:.0f} mg/dL** masih di atas 70, "
               f"namun pengklasifikasi kondisi menandai risiko **hipoglikemia** dalam {horizon_min} menit. "
               f"Pertimbangkan karbohidrat pencegahan & pantau ketat.")
elif lo95 is not None and lo95 < 70.0 <= pred:
    st.warning(f"🔻 **Ketidakpastian menyentuh zona hipoglikemia:** prediksi **{pred:.0f} mg/dL**, "
               f"tetapi batas bawah interval 95% mencapai **{lo95:.0f} mg/dL**. Pantau ketat.")
```

Dua pemicu, `elif` (tidak pernah keduanya sekaligus):
1. Pengklasifikasi kondisi menandai `hypoglycemia` **sementara** prediksi titik masih ≥ 70.
2. Batas bawah interval 95% (`pred − 3,3·σ`) turun di bawah 70 **sementara** prediksi titik masih ≥ 70.

**(c) Lapisan ketiga di dalam `PatientState`** — divergensi yang masuk ke kueri retrieval, bukan ke UI. Salinan persis ([patient_state.py:152-161](../src/patient_state.py#L152-L161)):
```python
# Kondisi yang perlu diantisipasi: kondisi terprediksi, DITAMBAH kondisi berisiko
# yang masih tercakup interval ketidakpastian meski prediksi titiknya normal.
conditions = [self.risk_level.replace("critical_", "")]
if self.predicted_lower is not None and self.predicted_lower < GLUCOSE_LOW:
    conditions.append("hypoglycemia")
if self.predicted_upper is not None and self.predicted_upper > GLUCOSE_HIGH:
    conditions.append("hyperglycemia")
# dahulukan kondisi berisiko, buang duplikat, pertahankan urutan
priority = {"hypoglycemia": 0, "hyperglycemia": 1, "normal": 2}
self.anticipated_conditions = sorted(set(conditions), key=lambda c: priority.get(c, 3))
```

**Namun lapisan (c) tidak aktif di aplikasi.** `predicted_lower`/`predicted_upper` sengaja **tidak** diteruskan ke `patient_state` — lihat komentar eksplisit di [streamlit_app.py:250-255](../app/streamlit_app.py#L250-L255):

> "Batas interval SENGAJA tidak diteruskan ke kueri. Memperluas kueri dengan semua kondisi yang tercakup interval memang menaikkan cakupan kondisi sebenarnya (94,2%), tetapi mengencerkan sinyal sehingga MRR justru turun ke 0,753 […] Interval tetap dipakai, namun sebagai PERINGATAN klinis kepada dokter."

Jadi dalam aplikasi, `anticipated_conditions` selalu berisi tepat satu elemen, dan blok "Interval prediksi … masih mencakup risiko" di [conditioned_query.py:134-138](../src/rag/conditioned_query.py#L134-L138) tidak pernah aktif.

### E.7 Konfirmasi: classifier terlatih atau rule-based?

**Keduanya — sistem berlapis. Pengklasifikasi terlatih menimpa hasil pengambangan regresi, kecuali untuk kondisi kritis.**

Logika penimpaan (salinan persis, [patient_state.py:142-150](../src/patient_state.py#L142-L150)):
```python
# Bila pengklasifikasi kondisi tersedia, ia MENGGANTIKAN kondisi hasil pengambangan
# nilai regresi — kecuali regresi sudah menandai kondisi kritis, yang tetap dihormati.
if self.predicted_condition and not self.risk_level.startswith("critical"):
    if self.predicted_condition == "hypoglycemia":
        self.risk_level, self.risk_label = "hypoglycemia", "BAHAYA - Hipoglikemia"
    elif self.predicted_condition == "hyperglycemia":
        self.risk_level, self.risk_label = "hyperglycemia", "HATI-HATI - Hiperglikemia"
    elif self.predicted_condition == "normal":
        self.risk_level, self.risk_label = "normal", "AMAN"
```

Jadi urutannya: (1) ambang rule-based 54/70/180/250 atas nilai regresi → (2) jika `predicted_condition` tersedia **dan** hasil (1) bukan kritis, kelas dari pengklasifikasi menang.

**Pengklasifikasi tersebut TERLATIH.** Rincian dari [scripts/train_condition_classifier.py](../scripts/train_condition_classifier.py):

| Aspek | Nilai | Lokasi |
|-------|-------|--------|
| **Algoritma** | `sklearn.ensemble.RandomForestClassifier` | [:113-119](../scripts/train_condition_classifier.py#L113-L119) |
| **Hyperparameter** | `n_estimators=200`, `max_depth=20`, `min_samples_split=5`, `random_state=42`, `n_jobs=-1` — dibaca dari `config.yaml model.random_forest` | [:114-118](../scripts/train_condition_classifier.py#L114-L118) |
| **`class_weight`** | `"balanced"` — inti usulan, agar kelas hipoglikemia tidak tenggelam | [:117](../scripts/train_condition_classifier.py#L117) |
| **Fitur** | **Identik dengan model regresi**: window `12 × 7` engineered features di-flatten menjadi **84 kolom** | [:63-66](../scripts/train_condition_classifier.py#L63-L66) |
| **Scaler** | **Dipakai ulang dari bundle regresi** (`bundle["scaler"]`), bukan di-fit ulang | [:111](../scripts/train_condition_classifier.py#L111) |
| **Horizon** | `HORIZON = 6` (+30 menit), hardcoded | [:42](../scripts/train_condition_classifier.py#L42) |
| **Split** | 10 latih / 2 uji (`patients[-2:]`) | [:61](../scripts/train_condition_classifier.py#L61) |
| **Keluaran** | `models/rf_condition_classifier_h6.pkl` | [:155-159](../scripts/train_condition_classifier.py#L155-L159) |

**Label** — tiga kelas, diturunkan dengan pengambangan **nilai glukosa sebenarnya** (bukan prediksi), salinan persis:
```python
# scripts/train_condition_classifier.py:44-48
CLASSES = ["hipoglikemia", "normal", "hiperglikemia"]

def classify_glucose(g: float) -> str:
    return "hipoglikemia" if g < 70 else "hiperglikemia" if g > 180 else "normal"

# scripts/train_condition_classifier.py:94-95
lbl_tr = np.array([classify_glucose(v) for v in ytr])
lbl_te = np.array([classify_glucose(v) for v in yte])
```

`ytr`/`yte` adalah glukosa **absolut** pada `t + 6 langkah` dari `create_sequences()`, sehingga label = kondisi sebenarnya 30 menit ke depan. Perhatikan: pengklasifikasi memakai ambang **70/180 saja** — tidak ada kelas "kritis".

**Pemetaan label Indonesia → Inggris** terjadi saat inferensi ([app/streamlit_app.py:59-60](../app/streamlit_app.py#L59-L60)):
```python
return {"hipoglikemia": "hypoglycemia", "normal": "normal",
        "hiperglikemia": "hyperglycemia"}.get(label, label)
```

**Hubungan dengan regresi — melengkapi, bukan menggantikan** ([train_condition_classifier.py:16-18](../scripts/train_condition_classifier.py#L16-L18)):

> "Model ini melengkapi — bukan menggantikan — model regresi: regresi tetap dipakai untuk menampilkan nilai prediksi dan intervalnya kepada dokter, sedangkan pengklasifikasi dipakai untuk (a) peringatan dini hipoglikemia dan (b) pengondisian kueri PC-RAG."

**Angka setelah segmentasi jendela (Tugas 5):**

> **BERUBAH (Tugas 5):** pengklasifikasi ikut dilatih ulang atas jendela tersegmentasi
> yang sama dengan regresi. Kalau hanya salah satu yang disaring, perbandingan
> "regresi lalu ambang" vs "pengklasifikasi" tidak lagi setara.

| Pendekatan | Metrik | Versi 1 | **Versi 2** |
|---|---|---|---|
| Regresi lalu ambang | akurasi keseluruhan | 88,8% | **89,3%** |
| | **sensitivitas hipoglikemia** | **14,0%** | **15,4%** |
| | PPV hipoglikemia | 57,1% | 52,4% |
| Pengklasifikasi 3 kelas | akurasi keseluruhan | 87,8% | **88,2%** |
| | **sensitivitas hipoglikemia** | **44,4%** | **45,2%** |
| | PPV hipoglikemia | 39,4% | **40,9%** |
| Pada kasus divergen | akurasi regresi | 35,9% | **36,5%** |
| | akurasi pengklasifikasi | 28,3% | **29,0%** |

Himpunan uji: 27.193 → 26.445 jendela; kasus divergen 3.621 → 3.416.
Durasi pelatihan pengklasifikasi: **122 detik**.

**Klaim inti tetap berlaku:** sensitivitas hipoglikemia naik ~3× (15,4% → 45,2%)
dengan mengorbankan PPV (52,4% → 40,9%).

**Keterbatasan yang juga tetap berlaku:** pada kasus divergen — justru kasus yang
menuntut antisipasi — pengklasifikasi **lebih buruk** daripada regresi (29,0% vs
36,5%). Ini bukan regresi baru; pola yang sama sudah ada pada Versi 1 (28,3% vs 35,9%).

Jadi rumusan yang akurat untuk Bab III/IV: **nilai** glukosa tetap dari regresi RF; **kondisi** yang membentuk kueri RAG dan badge risiko berasal dari pengklasifikasi RF terlatih; ambang rule-based bertindak sebagai lapisan dasar dan tetap berkuasa untuk kondisi kritis (<54 / >250 mg/dL).

Jalur aktivasi di aplikasi: `load_condition_classifier()` ([streamlit_app.py:37-48](../app/streamlit_app.py#L37-L48)) mengembalikan `None` jika berkas `.pkl` tak ada → `predict_condition()` mengembalikan `None` ([streamlit_app.py:52-53](../app/streamlit_app.py#L52-L53)) → sistem jatuh kembali ke ambang murni atas nilai regresi.

---

## F. PENGINDEKSAN KORPUS

> **BERUBAH TOTAL (Tugas 1):** korpus berpindah dari `additional_docs/` (14 PDF
> PERKENI/ADA) ke `books/` (12 dokumen KB-01..KB-12); ekstraksi menjadi per halaman
> sehingga chunk tidak lagi melintasi batas halaman; metadata halaman kini tersimpan
> dan berasal dari `manifest.csv`. Seluruh subbagian F berbeda dari Versi 1.

### F.1 Path skrip pengindeksan

Yang menghasilkan indeks produksi adalah `reingest_kb.py`.

| Path | Peran | Status |
|------|-------|--------|
| [scripts/reingest_kb.py](../scripts/reingest_kb.py) | **Jalur produksi.** Ingest `manual_kb.json` + PDF per halaman ke Chroma, rebuild dari nol | Menghasilkan indeks yang ada sekarang |
| [data/knowledge_base/manifest.csv](../data/knowledge_base/manifest.csv) | **Kontrak metadata.** Sumber tunggal kb_id, lembaga, tahun, judul, dan offset halaman | Wajib ada; ingest abort tanpanya |
| [scripts/build_eval_kb.py](../scripts/build_eval_kb.py) | Koleksi evaluasi RAGAS terpisah (2 halaman KB-03) | Tugas 6 |
| [scripts/ingest_kb.py](../scripts/ingest_kb.py) | CLI tipis, memanggil `RAGPipeline.ingest()` | Hanya `manual_kb.json`, **tanpa PDF** |
| [src/rag/knowledge_base.py](../src/rag/knowledge_base.py) | Kelas `MedicalKnowledgeBase`: chunking, metadata, persist | Dipakai semuanya |

**Korpus produksi** kini `data/knowledge_base/books/` — 12 dokumen:
KB-01 IDAI 2015 · KB-02/03 PERKENI 2021 · KB-04 PERKENI 2023 · KB-05 ADA-EASD 2021 ·
KB-06 ATTD 2019 · KB-07..KB-12 ISPAD 2022 (Ch10, 11, 12, 13, 14, 25).

`additional_docs/` sudah dipindahkan pengguna ke `data/knowledge_base/tidak terpakai lagi/`
dan tidak lagi dirujuk kode mana pun.

### F.2 Library ekstraksi PDF

**PyPDF2**, tetapi kini **per halaman**, bukan menggabungkan seluruh halaman:

```python
# scripts/reingest_kb.py — MENGGANTIKAN _read_pdf() lama
def _read_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    """Ekstrak PDF menjadi daftar (halaman_pdf 1-based, teks mentah)."""
    import PyPDF2

    pages: List[Tuple[int, str]] = []
    with path.open("rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for idx, page in enumerate(reader.pages, start=1):
            pages.append((idx, page.extract_text() or ""))
    return pages
```

> **BERUBAH (Tugas 1):** Versi 1 memakai `return "\n\n".join(pages)`. Penggabungan
> itulah yang menghancurkan batas halaman secara permanen dan membuat seluruh sitasi
> tampil `Hal. N/A`. Nomor halaman kini menjadi bagian dari kunci setiap dokumen.

Tidak ada pdfplumber, PyMuPDF/fitz, unstructured, atau OCR.

**Pembersihan teks** `_clean_text()` tidak berubah isinya, tetapi kini diterapkan
**per halaman**. Ini justru lebih tepat: header/footer jurnal adalah artefak per
halaman, sehingga penyaringan tidak lagi bocor antar halaman.

**Aturan abort** — seluruhnya fatal, tidak ada skip diam-diam:

1. PDF di `books/` tidak terdaftar di manifest → abort, seluruh nama berkas disebutkan.
2. Baris manifest tanpa berkas di disk → abort. Mengindeks 11 dari 12 dokumen secara
   senyap sama berbahayanya dengan mengindeks dokumen yang salah.
3. `hlm_total` ≠ jumlah halaman PDF sebenarnya → abort. Jumlah halaman yang meleset
   berarti offset diturunkan dari revisi berkas berbeda, sehingga **seluruh**
   `halaman_cetak` dokumen itu salah.
4. Total teks satu PDF < `--min-doc-chars` (300) → abort (indikasi hasil scan).

Kedua penjagaan pertama sudah diuji dan terbukti menghentikan proses.

### F.3 Strategi chunking

Fungsi: `MedicalKnowledgeBase.chunk_documents()` — [src/rag/knowledge_base.py:160-210](../src/rag/knowledge_base.py#L160-L210).

Salinan persis:
```python
# src/rag/knowledge_base.py:174-180
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
)
```

| Parameter | Nilai | Sumber |
|-----------|-------|--------|
| Splitter | `RecursiveCharacterTextSplitter` (langchain_text_splitters) | `chunk_documents()` |
| `chunk_size` | **900** karakter | **`config.yaml` → `rag.chunk_size`** |
| `chunk_overlap` | **120** karakter | **`config.yaml` → `rag.chunk_overlap`** |
| `separators` | `["\n## ", "\n### ", "\n\n", "\n", " ", ""]` | hardcoded |
| chunk manual_kb | **350/40** | **`config.yaml` → `rag.manual_kb.*`** |
| Filter fragmen | chunk < **80** karakter dibuang | `--min-chunk-chars` |

> **BERUBAH (Tugas 1 & 4):** Versi 1 mencatat `chunk_size`/`chunk_overlap` sebagai
> konstanta modul `_DEFAULT_CHUNK_SIZE`/`_DEFAULT_CHUNK_OVERLAP` yang membuat nilai
> di `config.yaml` **tidak dibaca**. Kedua konstanta itu sudah dihapus; config kini
> otoritatif. Nilai efektifnya kebetulan sama (900/120), jadi tidak ada perubahan
> perilaku — yang berubah adalah dari mana nilainya berasal.

**Keputusan arsitektur chunking per halaman.** `reingest_kb.py` memecah PDF menjadi
dokumen-per-halaman lalu menyuapkannya ke `chunk_documents()` yang **tidak berubah
logikanya**. Karena splitter sudah bekerja per dokumen, jaminan "tidak ada chunk
melintasi batas halaman" muncul otomatis tanpa logika baru, dan entri `manual_kb.json`
(yang tidak punya halaman) tetap jalan tanpa percabangan.

`doc_id` dibuat unik per halaman — `f"{kb_id}_p{halaman_pdf:04d}"` → `KB-09_p0001` —
karena `chunk_documents()` me-reset `part_idx` per dokumen; tanpa ini `chunk_id`
halaman 1 dan halaman 2 akan bertabrakan.

**Filter fragmen ekor.** Pemecahan per halaman menghasilkan potongan pendek di ujung
tiap halaman yang dulu tersembunyi oleh penggabungan antar halaman. Chunk di bawah
80 karakter dibuang; pada indeks saat ini **145 fragmen** tersaring.

**Fallback splitter** aktif jika `langchain_text_splitters` gagal di-import: jendela
**kata** (bukan karakter) dengan `step = chunk_size - overlap`.

### F.4 Daftar lengkap metadata per chunk

> **BERUBAH (Tugas 1) — INI PERBAIKAN TERPENTING DI SELURUH REFAKTOR.**
> Versi 1 mencatat: *"Nomor halaman TIDAK disimpan. Tidak untuk halaman PDF, tidak
> untuk halaman cetak."* Sekarang **disimpan keduanya**, beserta metadata bibliografis
> lengkap dari manifest.

**(a) Chunk dari dokumen pedoman (KB-01..KB-12)** — 12 field, seluruhnya skalar
(ChromaDB menolak list/dict/None). Dihasilkan `_page_metadata()`:

| Field | Contoh nilai aktual | Asal |
|-------|---------------------|------|
| `kb_id` | `KB-03` | manifest |
| `source_id` | `PERKENI-INSULIN-2021` | manifest |
| `nama_dokumen` | `KB-03_PERKENI-2021_Terapi-Insulin.pdf` | manifest (`nama_berkas`) |
| `lembaga` | `PERKENI` | manifest |
| `tahun` | `2021` (int) | manifest |
| `judul_lengkap` | `Pedoman Petunjuk Praktis Terapi Insulin ... 2021` | manifest |
| **`halaman_pdf`** | **`33`** (int) | posisi halaman di berkas |
| **`halaman_cetak`** | **`18`** (int; `0` = tidak valid) | `halaman_pdf + offset` |
| **`halaman_cetak_valid`** | **`True`** (bool) | `False` untuk front matter |
| `doc_id` | `KB-03_p0033` | `f"{kb_id}_p{halaman_pdf:04d}"` |
| `chunk_id` | `KB-03_p0033_ch_001` | `chunk_documents` |
| `domain` | `buku_panduan` | hardcoded |

**Rumus halaman:** `halaman_cetak = halaman_pdf + offset`, dengan `offset` dari
manifest. Nilainya bisa negatif (dokumen yang memulai penomoran ulang setelah front
matter; KB-03 = −15) atau besar (artikel jurnal berpenomoran berkelanjutan;
KB-09 = +1321).

**Halaman tanpa nomor cetak tidak memakai `None`** — ChromaDB menolaknya. Dipakai
pasangan sentinel `halaman_cetak=0` + `halaman_cetak_valid=False`. `_sanitize_metadata()`
juga diperkuat untuk membuang kunci bernilai `None`.

**Halaman depan tidak diindeks sama sekali.** Halaman dengan `halaman_cetak < 1`
adalah sampul dan daftar isi — padat kata kunci topik tanpa isi berguna, sehingga
kueri "hipoglikemia" bisa menarik baris daftar isi alih-alih tata laksananya.
Tersaring **45 dari 543 halaman**, seluruhnya pada empat dokumen Indonesia yang
berpenomoran ulang (KB-01 11, KB-02 13, KB-03 15, KB-04 6); dokumen ISPAD/ADA/ATTD
tidak kehilangan satu halaman pun.

**(b) Chunk dari `manual_kb.json`** — metadata penuh dari `_build_metadata()`, 24 field.
Tidak berubah dari Versi 1; `halaman` di sini tetap literal `"N/A"` karena entri kurasi
memang tidak punya halaman:

| Field | Contoh nilai aktual | Asal |
|-------|---------------------|------|
| `doc_id` | `manual_doc_0` | `_build_metadata` |
| `chunk_id` | `manual_doc_0_ch_000` | ditimpa di `chunk_documents` [:190](../src/rag/knowledge_base.py#L190) |
| `chunk_index` | `0` | [:189](../src/rag/knowledge_base.py#L189) |
| `domain` | `manual` | hardcoded |
| `subdomain` | `hiperglikemia` | `topic.lower().replace(" ","_")` |
| `sumber` | `Manual KB` | hardcoded |
| `judul` | `Hiperglikemia` | = `topic` |
| `tahun` | `2024` | hardcoded |
| `versi` | `v1` | hardcoded |
| `url_sumber` | `` (string kosong) | hardcoded |
| **`halaman`** | **`N/A`** | **hardcoded literal `"N/A"`** [:260](../src/rag/knowledge_base.py#L260) |
| `jenis_dm` | `dm_tipe2` | list `["dm_tipe2"]` → di-flatten ke string |
| `setting` | `fktp, fkrtl` | list → string |
| `sasaran` | `dokter_umum, sppd` | list → string |
| `populasi_khusus` | `umum` | list → string |
| `tipe_konten` | `panduan_klinis` | hardcoded |
| `bahasa` | `id` | hardcoded |
| `level_bukti` | `guideline` | hardcoded |
| `topik_terkait` | `Hiperglikemia` | list `[topic]` → string |
| `perlu_update_sebelum` | `2026-12` | hardcoded |
| `status` | `aktif` | hardcoded |
| `source` | `manual_kb` | dari dokumen |
| `created_at` | `2026-07-08T11:49:17.787477` | `datetime.utcnow()` saat ingest |
| `index` | `0` | urutan dokumen |
| `topic` | `Hiperglikemia` | ditambahkan di `save_to_chroma` [:320](../src/rag/knowledge_base.py#L320) |

**Jawaban tegas atas pertanyaan halaman (Versi 2):**

> **Nomor halaman DISIMPAN — keduanya.** `halaman_pdf` (posisi di berkas) dan
> `halaman_cetak` (nomor tercetak di halaman), plus penanda `halaman_cetak_valid`.

Pada Versi 1 jawabannya kebalikan: *"Nomor halaman TIDAK disimpan"* karena
`_read_pdf()` menggabungkan seluruh halaman menjadi satu string sebelum chunking.

**Verifikasi terhadap PDF sumber.** Lima chunk diperiksa, mencakup ketiga kasus
offset. Dua pemeriksaan per sampel: (A) teks chunk berada di dalam halaman PDF yang
diklaim; (B) teks chunk **tidak** muncul di halaman tetangga — bukti tidak melintasi
batas.

| Kasus offset | kb_id | halaman_pdf | offset | halaman_cetak | A | B |
|---|---|---|---|---|---|---|
| negatif | KB-02 | 22 | −13 | **9** | ya | tidak di tetangga |
| besar | KB-11 | 2 | +1340 | **1342** | ya | tidak di tetangga |
| nol | KB-07 | 15 | 0 | **15** | ya | tidak di tetangga |
| kontrol | KB-04 | 85 | −6 | **79** | ya | tidak di tetangga |
| kontrol | KB-12 | 10 | +1528 | **1538** | ya | tidak di tetangga |

**Konfirmasi independen.** Pada dokumen Indonesia, nomor cetak ikut terekstrak ke
dalam teks halaman dan cocok dengan nilai yang dihitung:

- KB-02 hal. 9 → teks halaman diawali `"9 Pedoman Pemantauan Glukosa Darah Mandiri - 2021 Tabel 3..."`
- KB-04 hal. 79 → diawali `"79 Untuk menentukan pemberian insulin, pemeriksaan GD, HbA1c..."`
- KB-04 hal. 64 → diawali `"64 Tabel 1. Perbedaan Antar Protokol Penggunaan Insulin Intravena..."`
- KB-04 hal. 41 → diawali `"41 Manajemen Periode Intra-Operatif..."`

**Catatan KB-05.** Berkas ini versi *Online First* Springer yang tidak mencetak nomor
halaman sama sekali; offset 2608 diturunkan dari rentang terbitan *Diabetologia*
64(12):2609–2652. Verifikasinya lewat pencocokan isi, bukan pencarian angka —
ketiadaan nomor di halaman bukan kegagalan verifikasi.

Sitasi yang kini terbentuk di antarmuka: `PERKENI (2021) · Hal. 9 · KB-02_...pdf`
(lihat G.6), menggantikan `[nama-berkas.pdf, N/A, Hal. N/A]`.

### F.5 Persistensi ChromaDB dan pengecekan ekstraksi ulang

**Persistensi:** `langchain_chroma.Chroma` dengan `persist_directory`, di `save_to_chroma()` — salinan persis ([knowledge_base.py:298-330](../src/rag/knowledge_base.py#L298-L330)):
```python
vector_store = Chroma(
    collection_name=self.collection_name,
    embedding_function=embeddings,
    persist_directory=str(self.persist_dir),
)

if reset_collection:
    try:
        vector_store.delete_collection()
        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=str(self.persist_dir),
        )
    except Exception as exc:
        logger.warning("Could not reset existing collection: %s", exc)
...
vector_store.add_documents(documents)
```

Lokasi persist: `models/chroma_db/` (berisi `chroma.sqlite3` + direktori segmen HNSW). Default di [pipeline.py:58](../src/rag/pipeline.py#L58) dan [reingest_kb.py:86](../scripts/reingest_kb.py#L86).

**Apakah ada pengecekan agar ekstraksi tidak diulang setiap kali aplikasi dijalankan?**

Jawabannya berlapis, dan yang penting untuk laporan adalah lapisan pertama:

**(1) Aplikasi tidak pernah melakukan ekstraksi atau ingest sama sekali.** `load_rag()` hanya menyambung ke koleksi yang sudah ada — salinan persis ([app/streamlit_app.py:78-87](../app/streamlit_app.py#L78-L87)):
```python
@st.cache_resource
def load_rag():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    p = RAGPipeline()          # seluruh parameter dibaca dari config.yaml (Tugas 4)
    p.build()
    return p
```

> **BERUBAH (Tugas 4):** Versi 1 memuat `kb_dir` hardcoded dan dua `os.getenv` inline
> di dalam `load_rag()`. Semuanya dihapus; `RAGPipeline()` meresolusi sendiri dari
> `config.yaml`.

`p.build()` ([pipeline.py:135-158](../src/rag/pipeline.py#L135-L158)) hanya mengkonstruksi `MMRRetriever`, yang membuka koleksi Chroma yang sudah persist. `p.ingest()` **tidak dipanggil**. Dekorator `@st.cache_resource` membuat objek pipeline dibangun sekali per sesi Streamlit, bukan per interaksi.

Jadi: **ekstraksi PDF tidak pernah diulang saat aplikasi dijalankan**, karena aplikasi memang tidak punya jalur ekstraksi. Indexing adalah langkah CLI manual yang terpisah.

**(2) Skrip ingest sendiri justru sengaja TIDAK idempoten-dengan-lewat; ia rebuild dari nol.** Salinan persis ([reingest_kb.py:133-140](../scripts/reingest_kb.py#L133-L140)):
```python
# 3) Rebuild Chroma dari nol (idempoten) — hapus folder persist yang mungkin korup/versi lama
persist = Path(args.persist)
if persist.exists():
    shutil.rmtree(persist, ignore_errors=True)
    print(f"[3] Folder chroma lama dihapus (fresh rebuild): {persist}")

chunks = kb.chunk_documents(documents=docs)
ok = kb.save_to_chroma(chunks=chunks, reset_collection=False)
```

"Idempoten" di sini berarti *hasilnya deterministik setiap run*, **bukan** *melewati pekerjaan yang sudah dilakukan*. Setiap eksekusi `python scripts/reingest_kb.py` menghapus seluruh `models/chroma_db/`, mengekstrak ulang ke-12 dokumen, dan meng-embed ulang seluruh chunk. **Tidak ada cache ekstraksi, tidak ada pemeriksaan mtime, tidak ada pemeriksaan hash isi.**

> **BERUBAH (Tugas 1):** `save_to_chroma()` kini juga menetapkan
> `collection_metadata={"hnsw:space": "cosine"}`. Koleksi lama memakai default **l2**,
> yang membuat skor relevansi LangChain (`1 − d/√2`) bisa bernilai negatif. Dengan
> embedding yang sudah dinormalisasi, cosine memberi skor di `[0,1]` yang dapat
> ditampilkan langsung ke dokter (lihat G.2).

**(3) Satu-satunya deduplikasi berbasis hash ada di jalur lain** yang tidak dipakai produksi — `import_kb_docs.py`, salinan persis:
```python
# scripts/import_kb_docs.py:96-99
def _doc_id_for(path: Path, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", path.stem).strip("_").lower() or "doc"
    return f"ext_{stem}_{digest}"

# scripts/import_kb_docs.py:144-147
doc_id = _doc_id_for(path, text)
if doc_id in existing_ids:
    skipped.append(f"{path.name} (duplicate)")
    continue
```

Ini melewatkan dokumen yang isinya identik — tetapi PDF tetap **harus diekstrak lebih dulu** untuk menghitung hash-nya ([import_kb_docs.py:138-144](../scripts/import_kb_docs.py#L138-L144)), jadi ekstraksi tidak dihemat. Dan skrip ini hanya menulis ke `manual_kb.json`, bukan ke Chroma.

**Verifikasi jumlah** ([reingest_kb.py:150-155](../scripts/reingest_kb.py#L150-L155)) membuka ulang koleksi dan membandingkan `col.count()` dengan `len(chunks)`, mencetak `OK` atau `MISMATCH!`.

### F.6 Jumlah chunk dan dokumen saat ini

Dibaca langsung dari `models/chroma_db/chroma.sqlite3` (koleksi `diabetes_kb`, id `aecfb677-00a8-4e22-998f-91f93b1316d6`):

> **BERUBAH (Tugas 1):** korpus, jumlah dokumen, dan jumlah chunk seluruhnya berbeda.
> Versi 1 mencatat 2.585 chunk dari 21 dokumen (14 PDF PERKENI/ADA + 7 entri kurasi).

**Total: 2.061 chunk dari 19 dokumen** (12 dokumen pedoman + 7 entri `manual_kb.json`),
dibangun dari **494 halaman terindeks**.

| Sumber | Chunk |
|--------|-------|
| KB-05_ADA-EASD-2021_Tatalaksana-DMT1-Dewasa.pdf | 278 |
| KB-04_PERKENI-2023_Tatalaksana-Hiperglikemia-RS.pdf | 255 |
| KB-01_IDAI-2015_Konsensus-DMT1.pdf | 249 |
| KB-11_ISPAD-2022_Ch14-Aktivitas-Fisik.pdf | 219 |
| KB-07_ISPAD-2022_Ch10-Nutrisi.pdf | 192 |
| KB-08_ISPAD-2022_Ch11-KAD-HHS.pdf | 164 |
| KB-12_ISPAD-2022_Ch25-Sumber-Daya-Terbatas.pdf | 162 |
| KB-03_PERKENI-2021_Terapi-Insulin.pdf | 144 |
| KB-09_ISPAD-2022_Ch12-Hipoglikemia.pdf | 142 |
| KB-10_ISPAD-2022_Ch13-Manajemen-Hari-Sakit.pdf | 87 |
| KB-06_ATTD-2019_Target-Glikemik-TIR.pdf | 86 |
| KB-02_PERKENI-2021_Pemantauan-Glukosa-Mandiri.pdf | 69 |
| `manual_kb` (7 entri kurasi) | 14 |
| **Total** | **2.061** |

**Neraca halaman:**

| | Halaman |
|---|---|
| Total halaman PDF (12 dokumen) | 543 |
| Front matter tersaring (`halaman_cetak < 1`) | −45 |
| Halaman terlalu pendek (< 100 karakter) | −4 |
| **Terindeks** | **494** |

Ditambah 145 fragmen chunk < 80 karakter yang dibuang pasca-chunking.

Komposisi korpus: **6 chapter ISPAD 2022** (Ch10, 11, 12, 13, 14, 25) + **3 dokumen
PERKENI** (2021 PGDM, 2021 Terapi Insulin, 2023 Hiperglikemia RS) + **IDAI 2015** +
**ADA-EASD 2021** + **ATTD 2019** + KB kurasi manual.

**Catatan komposisi bahasa yang relevan untuk retrieval:** delapan dari dua belas
dokumen berbahasa Inggris (seluruh ISPAD, ADA-EASD, ATTD), sedangkan kueri sistem
dibangun dalam Bahasa Indonesia. Ketidakcocokan ini menekan skor kemiripan dan
terkuantifikasi pada Tugas 7 (lihat lampiran).

### F.7 Model embedding

**Nama persis: `all-MiniLM-L6-v2`** (Sentence-Transformers / HuggingFace).
**Dimensi: 384** — diverifikasi langsung dari kolom `dimension` pada tabel `collections` di `chroma.sqlite3`.

> **BERUBAH (Tugas 4):** nama model kini dibaca dari `config.yaml` → `rag.embedding_model`
> melalui `RagConfig`, bukan dari konstanta modul. Nilai efektifnya tetap
> `all-MiniLM-L6-v2`, jadi indeksnya sebanding; yang berubah adalah asal nilainya.
>
> Lebih penting: Versi 1 mencatat `_HF_EMBED_MODEL` **terduplikasi** di
> `knowledge_base.py` dan `retriever.py`, keduanya dibaca saat import. Kalau hanya
> satu yang diubah, embedding dokumen dan embedding kueri dihasilkan model berbeda
> dan retrieval merosot menjadi derau **tanpa error apa pun**. Kini `_build_embeddings`
> hanya ada **satu implementasi** yang dipakai bersama, dijaga tes
> `test_embedding_model_shared_between_ingest_and_query`.

Konstruksi ([knowledge_base.py:29-36](../src/rag/knowledge_base.py#L29-L36)):
```python
if embed_provider == "sentence-transformers":
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=_HF_EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
```

Catatan: `normalize_embeddings=True` — vektor dinormalisasi ke norma 1, sehingga jarak yang dipakai Chroma setara dengan kesamaan kosinus. Berjalan **di CPU**.

Provider embedding alternatif yang tersedia ([knowledge_base.py:23-49](../src/rag/knowledge_base.py#L23-L49)):
- `"google"` → `GoogleGenerativeAIEmbeddings(model="models/embedding-001")`
- `"ollama"` → `OllamaEmbeddings(model="nomic-embed-text")` (default parameter, tetapi bukan default provider)

Provider aktif dari `.env`: `EMBED_PROVIDER=sentence-transformers`.
`config.yaml:80` juga menyebut `embedding_model: "all-MiniLM-L6-v2"`, tetapi nilai itu **tidak dibaca kode mana pun** — yang berlaku adalah konstanta modul / env `HF_EMBED_MODEL`.

Direktori `models/chroma_db_multilingual/` juga ada (dari eksperimen `scripts/eval_embedding_alternative.py`), tetapi bukan indeks produksi.

---

## G. PENELUSURAN DAN PEMBANGKITAN

### G.1 Fungsi penyusun kueri dari objek kondisi klinis

**Path:** [src/rag/conditioned_query.py](../src/rag/conditioned_query.py)
**Kelas:** `PredictionConditionedQueryBuilder` — [:64-239](../src/rag/conditioned_query.py#L64-L239)
**Method utama:** `build()` [:78-96](../src/rag/conditioned_query.py#L78-L96) → `_primary_query()` [:100-151](../src/rag/conditioned_query.py#L100-L151)

Method pendukung:

| Method | Lokasi | Keluaran |
|--------|--------|----------|
| `_primary_query()` | [:100-151](../src/rag/conditioned_query.py#L100-L151) | String kueri untuk ChromaDB |
| `_llm_context()` | [:153-184](../src/rag/conditioned_query.py#L153-L184) | Blok konteks untuk prompt LLM |
| `_contributing_factors()` | [:186-205](../src/rag/conditioned_query.py#L186-L205) | Daftar faktor klinis aktif |
| `_strategy_question()` | [:207-230](../src/rag/conditioned_query.py#L207-L230) | Sufiks pertanyaan sesuai strategi |
| `_metadata_tags()` | [:232-239](../src/rag/conditioned_query.py#L232-L239) | Hint filter metadata |

Titik masuk dari pipeline: `RAGPipeline._build_query()` [pipeline.py:239-287](../src/rag/pipeline.py#L239-L287).

**Template kueri — `_primary_query()`, salinan persis:**

```python
# src/rag/conditioned_query.py:109-151
parts: List[str] = []

# 1. Prediction-conditioned core statement
parts.append(
    f"Prediksi glukosa {state.prediction_horizon_minutes} menit ke depan: "
    f"{state.predicted_glucose:.1f} mg/dL "
    f"(dari {state.current_glucose:.1f} mg/dL, "
    f"perubahan {state.glucose_delta:+.1f} mg/dL, tren {state.trend_label})."
)

# 2. Clinical risk classification
parts.append(f"Status risiko prediksi: {state.risk_label}.")

# 2b. Pengondisian sadar-ketidakpastian: kondisi berisiko yang masih tercakup
# interval prediksi tetap dimunculkan pada kueri meski prediksi TITIK-nya normal,
# agar retrieval tidak buta terhadap bahaya yang mungkin terjadi (lihat Bab VI).
risk_terms = {
    "hypoglycemia": "hipoglikemia (glukosa di bawah 70 mg/dL)",
    "hyperglycemia": "hiperglikemia (glukosa di atas 180 mg/dL)",
}
extra = [
    risk_terms[c]
    for c in state.anticipated_conditions
    if c in risk_terms and c != state.risk_level.replace("critical_", "")
]
if extra and state.predicted_lower is not None and state.predicted_upper is not None:
    parts.append(
        f"Interval prediksi {state.predicted_lower:.0f}-{state.predicted_upper:.0f} mg/dL "
        f"masih mencakup risiko {', '.join(extra)}; sertakan penanganannya."
    )

# 3. Contributing factors (ordered by clinical significance)
factors = self._contributing_factors(state)
if factors:
    parts.append(f"Faktor kontribusi: {', '.join(factors)}.")

# 4. Strategy-specific question
if user_question:
    parts.append(user_question)
else:
    parts.append(self._strategy_question(state))

return " ".join(parts)
```

**Sufiks strategi COMPREHENSIVE** (default dan satu-satunya yang dipakai produksi — [pipeline.py:267](../src/rag/pipeline.py#L267)), salinan persis:
```python
# src/rag/conditioned_query.py:225-230
return (
    f"Berikan penilaian risiko, tindakan pencegahan, dan protokol pemantauan "
    f"untuk kondisi prediksi glukosa {state.predicted_glucose:.0f} mg/dL "
    f"({state.risk_label}) dalam {state.prediction_horizon_minutes} menit ke depan. "
    "Sertakan rekomendasi yang bisa dilakukan dokter maupun pasien."
)
```

Tiga strategi lain (`RISK_FOCUSED`, `INTERVENTION_FOCUSED`, `MONITORING_FOCUSED`, [:209-223](../src/rag/conditioned_query.py#L209-L223)) terdefinisi tetapi **tidak pernah dipilih** di jalur produksi.

**Ambang faktor kontribusi** ([:186-205](../src/rag/conditioned_query.py#L186-L205)):

| Faktor | Ambang | Teks |
|--------|--------|------|
| tren cepat | `trend_rate == "rapid"` | "tren cepat meningkat" / "tren cepat menurun" |
| insulin aktif | `IOB ≥ 1.0` | "insulin aktif {…} unit" |
| karbohidrat | `COB ≥ 10.0` | "karbohidrat belum terserap {…} g" |
| stres | `stress_level ≥ 7` | "stres tinggi ({…}/10)" |
| aktivitas rendah | `activity_level < 15` | "aktivitas fisik rendah" |
| aktivitas tinggi | `activity_level ≥ 60` | "aktivitas tinggi ({…} menit)" |

**Contoh kueri terbentuk** (kasus divergen, current 150 → prediksi 214, horizon 30 menit):
```
Prediksi glukosa 30 menit ke depan: 214.0 mg/dL (dari 150.0 mg/dL, perubahan
+64.0 mg/dL, tren meningkat). Status risiko prediksi: HATI-HATI - Hiperglikemia.
Faktor kontribusi: tren cepat meningkat, aktivitas fisik rendah. Berikan penilaian
risiko, tindakan pencegahan, dan protokol pemantauan untuk kondisi prediksi glukosa
214 mg/dL (HATI-HATI - Hiperglikemia) dalam 30 menit ke depan. Sertakan rekomendasi
yang bisa dilakukan dokter maupun pasien.
```

**Blok "Interval prediksi …" (bagian 2b) tidak pernah muncul di aplikasi**, karena `predicted_lower`/`predicted_upper` sengaja tidak diteruskan — lihat E.6(c).

**Ekspansi kueri kedua.** Setelah `_primary_query()`, `MMRRetriever.retrieve_with_context()` menambahkan sufiks lagi lewat `_enhance_query()`:
```python
def _enhance_query(self, query: str, patient_state: Dict[str, Any]) -> str:
    from src.constants import CLASS_NORMAL, classify_glucose_3class

    tags: List[str] = []
    glucose = float(patient_state.get("current_glucose", 0.0))
    stress = int(patient_state.get("stress_level", 0))

    kondisi = classify_glucose_3class(glucose)
    if kondisi != CLASS_NORMAL:
        tags.append(kondisi)

    if stress >= 7:
        tags.append("stress tinggi")

    if not tags:
        return query
    return f"{query}. Konteks pasien: {', '.join(tags)}."
```

Perhatikan bahwa lapisan ini memakai **`current_glucose`**, bukan prediksi — jadi ia menarik kueri kembali ke kondisi saat ini. Pada kasus divergen (current normal), `tags` kosong dan kueri tidak berubah.

**Metadata filter tidak dipakai.** `_metadata_tags()` menghasilkan dict berisi `{"jenis_dm": "dm_tipe2", "topik": …}` ([:232-239](../src/rag/conditioned_query.py#L232-L239)), tetapi `ConditionedQuery.to_pipeline_kwargs()` ([:265-271](../src/rag/conditioned_query.py#L265-L271)) tidak menyertakannya, dan `RAGPipeline._retrieve()` ([pipeline.py:234-237](../src/rag/pipeline.py#L234-L237)) memanggil retriever tanpa `metadata_filter`. Jadi **retrieval berjalan tanpa filter metadata apa pun** — konsisten dengan fakta bahwa chunk PDF memang tidak punya field `jenis_dm`/`topik`.

### G.2 Parameter retrieval

> **BERUBAH (Tugas 1B & 4):** parameter kini dibaca dari `config.yaml`, dan skor
> kemiripan yang dulu selalu `0.0` palsu kini nyata.

Salinan persis `MMRRetriever.retrieve()`:
```python
top_k = top_k if top_k is not None else self.top_k
fetch_k = self.fetch_k
lambda_mult = self.lambda_mult

if self._embeddings is not None:
    qvec = self._embeddings.embed_query(query)              # embed SEKALI
    pool = self._vector_store.similarity_search_by_vector_with_relevance_scores(
        embedding=qvec, k=fetch_k, filter=metadata_filter
    )
    score_by_key = {self._doc_key(d): float(s) for d, s in pool}
    docs = self._vector_store.max_marginal_relevance_search_by_vector(
        embedding=qvec, k=top_k, fetch_k=fetch_k,
        lambda_mult=lambda_mult, filter=metadata_filter,
    )
```

| Parameter | Nilai efektif | Sumber |
|-----------|---------------|--------|
| `search_type` | **MMR** (Maximal Marginal Relevance) | `max_marginal_relevance_search_by_vector` |
| `k` | **4** | **`config.yaml` → `rag.top_k_retrieval`** |
| `fetch_k` | **12** | **`config.yaml` → `rag.fetch_k`** |
| `lambda_mult` | **0.5** | **`config.yaml` → `rag.lambda_mult`** |
| `filter` | tidak diberikan (lihat G.1) | — |

`lambda_mult = 0.5` = bobot seimbang antara relevansi dan keragaman (1,0 = relevansi murni, 0,0 = keragaman maksimum).

**Skor kemiripan kini nyata.** Versi 1 mengisi `"similarity": 0.0` untuk semua hasil
karena `as_retriever().invoke()` tidak mengembalikan skor. Sekarang vektor kueri
dihitung **sekali** lalu dipakai ulang oleh dua pemanggilan: satu untuk penilaian
(`k=fetch_k`) dan satu untuk seleksi MMR (`k=top_k`). Join dijamin total karena MMR
memilih dari kolam kandidat `fetch_k` yang sama.

Biaya tambahan: **satu kueri HNSW, nol panggilan embedding.** Seleksi MMR tidak
berubah sedikit pun, sehingga angka retrieval tetap sebanding dengan sebelumnya.

`max_marginal_relevance_search_with_score_by_vector` **tidak tersedia** di
`langchain-chroma 0.2.4` — opsi itu gugur karena absen, bukan preferensi.

**Caveat yang wajib ditampilkan ke dokter.** Skor adalah relevansi kueri-terhadap-chunk,
**bukan** objektif MMR (yang mengurangi penalti redundansi). Urutan rank karena itu
tidak selalu menurun monoton — terbukti pada hasil uji: rank 1 = 0,17 sementara
rank 3 dan 4 = 0,20. Tanpa keterangan ini, orang akan mengira itu bug dan
"memperbaikinya" dengan menyortir ulang.

**Tipe berubah:** `similarity` menjadi `Optional[float]`; `0.0` palsu dihapus karena
menyesatkan sebagai "kemiripan nol" padahal artinya "tidak diukur".

**Fallback retriever** `SimpleKeywordRetriever` ([retriever.py:51-88](../src/rag/retriever.py#L51-L88)) memakai skor overlap token: `overlap / max(len(query_tokens), 1)` ([:73-74](../src/rag/retriever.py#L73-L74)).

### G.3 Reranking

**TIDAK DITEMUKAN.** Pencarian `rerank`, `cross.encoder`, `CrossEncoder`, `cohere` di seluruh berkas `*.py` proyek mengembalikan **nol hasil**.

Pipeline bersifat satu tahap: MMR retrieval → langsung ke prompt. Diversifikasi hasil hanya lewat MMR (`fetch_k=12` → `k=4`, `lambda_mult=0.5`), bukan lewat model reranking terpisah.

### G.4 Model LLM dan parameter generatif

**Nama model persis: `gemini-2.5-flash-lite`**, berasal dari `.env`:
```
GEMINI_MODEL=gemini-2.5-flash-lite
```

> **BERUBAH (Tugas 4):** `temperature` kini 0,2 (dari config) bukan 0,1 (hardcoded),
> dan `max_tokens` **benar-benar diterapkan** — sebelumnya diterima lalu dibuang
> dengan `del` tanpa pernah sampai ke LLM.

Rantai resolusi kini melalui `RagConfig` dengan presedensi
**ctor > env > config.yaml > default** (lihat bagian I).

Konstruksi LLM — salinan persis:
```python
if provider == "gemini":
    api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")
    ...
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=self.temperature,
        max_output_tokens=self.max_tokens,
    )
else:
    llm = ChatOllama(
        model=model_name,
        base_url=self.ollama_base_url,
        temperature=self.temperature,
        num_predict=self.max_tokens,
    )
```

| Parameter | Versi 1 | **Versi 2** | Sumber |
|-----------|---------|-------------|--------|
| `model` | `gemini-2.5-flash-lite` | `gemini-2.5-flash-lite` | `.env` (`GEMINI_MODEL`) |
| `temperature` | 0.1 (hardcoded) | **0.2** | `config.yaml` → `rag.llm.temperature` |
| `max_tokens` | tidak diterapkan | **700** | `config.yaml` → `rag.llm.max_tokens` |
| `top_p` | TIDAK DITEMUKAN | **TIDAK DITEMUKAN** | tidak pernah di-set |
| `top_k` (sampling) | TIDAK DITEMUKAN | **TIDAK DITEMUKAN** | tidak pernah di-set |

**Dua nilai yang berubah perilakunya, bukan sekadar sumbernya:**

1. **`temperature` 0,1 → 0,2.** Config selalu menulis 0,2; kode memakai 0,1. Kini
   config yang menang, sesuai yang didokumentasikan laporan.
2. **`max_tokens` mulai berlaku, dan dinaikkan 300 → 700.** Menyalakannya pada 300
   akan **mulai memotong** keluaran; 300 token terlalu ketat untuk nasihat klinis
   Bahasa Indonesia beserta disclaimer.

**Kuota model (diverifikasi ulang 2026-08).** `gemini-2.5-flash-lite` tier gratis:
**15 RPM / 250.000 TPM / 1.000 RPD.** Komentar `config.yaml` versi lama menyatakan
"SEMUA model free = 20 request/HARI + 10/menit" — itu **keliru sekitar 50×** dan
sudah diperbaiki. Halaman resmi Google kini tidak lagi memuat tabel per model dan
mengarahkan ke dasbor AI Studio, jadi angka ini perlu dikonfirmasi sekali di dasbor
akun yang dipakai.

**Ketidaksesuaian dengan config yang harus akurat di laporan.** `config.yaml:91-92` menetapkan `temperature: 0.2` dan `max_tokens: 300`, tetapi:
- `temperature` efektif adalah **0.1** (hardcoded), bukan 0.2.
- `max_tokens` **tidak diterapkan sama sekali**. Satu-satunya tempat parameter itu muncul adalah `RAGGenerator.generate_explanation()`, di mana keduanya **dibuang secara eksplisit** — salinan persis ([generator.py:45-54](../src/rag/generator.py#L45-L54)):
  ```python
  def generate_explanation(
      self,
      context_docs: List[str],
      patient_state: Dict[str, Any],
      prediction: float,
      temperature: float = 0.1,
      max_tokens: int = 300,
  ) -> str:
      del temperature
      del max_tokens
  ```
  Terlebih lagi, `generate_explanation()` **tidak dipanggil** dari jalur produksi; `RAGPipeline.answer()` memanggil `generate_advisory()` ([pipeline.py:193](../src/rag/pipeline.py#L193)).

Rantai LCEL ([advisor_chain.py:76](../src/rag/advisor_chain.py#L76)): `prompt | llm | StrOutputParser()`.

### G.5 System prompt dan template prompt

**System prompt** — salinan persis ([src/rag/prompts.py:7-15](../src/rag/prompts.py#L7-L15)):

> **BERUBAH (Tugas 1B):** aturan 2 dan 3 baru. Model kini **dilarang menulis nomor
> halaman sendiri** dan wajib menyatakan bila konteksnya kosong.

```
Anda adalah asisten klinis berbasis panduan medis Indonesia untuk mendukung keputusan dokter dalam penanganan diabetes.

Aturan:
1. Jawab berdasarkan konteks yang diberikan.
2. JANGAN menulis nomor halaman, nomor bab, nomor tabel, atau tautan.
   Rujuk sumber HANYA dengan penanda [S1], [S2], ... sesuai nomor blok konteks.
   Nomor halaman ditampilkan oleh sistem dari metadata dokumen, bukan oleh Anda.
3. Jika blok konteks kosong, nyatakan secara eksplisit bahwa tidak ada rujukan
   panduan yang relevan pada knowledge base, dan JANGAN mengarang rujukan.
4. Jika konteks tidak cukup, katakan informasi belum tersedia pada knowledge base saat ini.
5. Untuk kondisi berisiko tinggi, sarankan evaluasi dokter segera.
6. Gunakan Bahasa Indonesia yang ringkas, jelas, dan actionable.
```

**Struktur ChatPromptTemplate** — salinan persis ([advisor_chain.py:41-51](../src/rag/advisor_chain.py#L41-L51)):
```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Konteks dokumen:\n{context}\n\n"
            "Data dan pertanyaan:\n{question_payload}\n\n"
            "Berikan jawaban klinis ringkas dengan langkah aksi dan disclaimer dokter.",
        ),
    ]
)
```

**Isi `{question_payload}`** — salinan persis ([prompts.py:52-59](../src/rag/prompts.py#L52-L59)):
```python
return (
    f"Pertanyaan klinisi: {query}\n"
    f"Data pasien: glukosa={gluc} mg/dL, "
    f"stress={stress}/10, "
    f"aktivitas={activity} menit, "
    f"insulin_on_board={insulin} unit, "
    f"carbs_on_board={carbs} gram.\n"
    f"Prediksi 1 jam: {pred_glucose} mg/dL, Risk: {pred_risk}."
)
```

Dua catatan akurasi pada payload ini:
1. Teks **"Prediksi 1 jam"** di-hardcode ([prompts.py:59](../src/rag/prompts.py#L59)), padahal horizon produksi adalah **+30 menit** (`prediction_horizon: 6`). Payload memberi tahu LLM horizon yang salah. Kueri retrieval (`{query}`) menyebut horizon yang benar, jadi prompt memuat dua horizon yang bertentangan.
2. `pred_risk` selalu `'N/A'` bila `prediction` berupa angka ([prompts.py:41-43](../src/rag/prompts.py#L41-L43)) — dan pipeline memang selalu mengirim float ([pipeline.py:204](../src/rag/pipeline.py#L204)). Jalur `dict` yang mengisi `risk_level` tidak pernah terpakai.

**Blok `{context}`** — dibentuk `format_context_with_citations()`, lihat G.6.

**Catatan struktural:** `_llm_context()` di [conditioned_query.py:153-184](../src/rag/conditioned_query.py#L153-L184) membangun blok konteks kaya berlabel `=== KONTEKS PREDIKSI GLUKOSA (PREDICTION-CONDITIONED RAG) ===` berisi urgensi, tren, IOB/COB. Blok ini disimpan di `ConditionedQuery.llm_context`, tetapi `to_pipeline_kwargs()` ([:265-271](../src/rag/conditioned_query.py#L265-L271)) **tidak menyertakannya** dan `RAGPipeline` tidak pernah membacanya. **Blok ini tidak pernah sampai ke LLM.** Yang sampai ke LLM hanyalah `primary_query` (sebagai `{query}` di dalam `question_payload`).

### G.6 Penyisipan sitasi dokumen dan halaman

> **BERUBAH (Tugas 1B) — DUA LAPIS.** Nomor halaman **dihapus dari blok konteks LLM**,
> lalu ditampilkan ke dokter dari metadata chunk. Versi 1 melakukan kebalikannya:
> mengirim nomor halaman ke LLM (yang nilainya `N/A`) dan tidak menampilkannya di UI.

**Lapis 1 — LLM tidak lagi melihat nomor halaman.** Salinan persis:
```python
def format_context_with_citations(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Nomor halaman SENGAJA TIDAK dimasukkan ke blok konteks. Aturan pada system
    prompt saja hanyalah jaminan lunak; tidak memberikan angkanya sama sekali adalah
    jaminan keras — model tidak dapat menyalin yang tak pernah dilihatnya."""
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
```

Penanda konteks kini `[S1] {judul} — {lembaga}, {tahun}` — **tanpa nomor halaman**.
Diverifikasi: penyebutan nomor halaman di dalam blok konteks = **0**.

Aturan prompt saja adalah jaminan lunak; tidak memberikan angkanya sama sekali adalah
jaminan keras. Butir inilah yang benar-benar memenuhi persyaratan "nomor halaman tidak
boleh berasal dari LLM".

**Lapis 2 — dokter melihat halaman dari metadata.** Modul baru
[src/rag/citations.py](../src/rag/citations.py) menjadi satu sumber kebenaran aturan
halaman, dipakai tiga konsumen (expander Streamlit, log keputusan, `_extract_sources`):

```python
PAGE_UNKNOWN = "Hal. tidak tercatat"

def format_page_label(meta) -> str:
    # halaman bercetak valid   -> "Hal. 1322"
    # front matter (offset < 1) -> "Hal. berkas 3 (bagian depan, tanpa nomor cetak)"
    # tanpa info halaman        -> "Hal. tidak tercatat"

def build_source_list(retrieved_docs, snippet_chars=200) -> List[Dict]
```

**Tampilan di UI** — expander tertutup `Sumber Rujukan (N dokumen)`:
```python
sources = build_source_list(res.get("retrieved_docs", []), snippet_chars=200)
st.session_state["last_rec_sources"] = sources

if not sources:
    st.error("**Tidak ditemukan rujukan panduan yang relevan** ... "
             "Teks di bawah TIDAK didukung kutipan panduan.")

st.markdown(f'<div class="card">{res["explanation"]}</div>', unsafe_allow_html=True)
...
if sources:
    st.caption("Isi rekomendasi bersumber dari dokumen pedoman yang tercantum pada "
               "**Sumber Rujukan** di bawah. Keputusan akhir berada pada dokter.")
    with st.expander(f"📚 Sumber Rujukan ({len(sources)} dokumen)", expanded=False):
        ...
```

Contoh baris yang tampil (hasil uji nyata, `provider=template` tanpa panggilan LLM):
```
#2 · Tatalaksana Pasien dengan Hiperglikemia di Rumah Sakit · kemiripan 0.18
   PERKENI (2023) · Hal. 64 · KB-04_PERKENI-2023_Tatalaksana-Hiperglikemia-RS.pdf
   Hipoglikemia (<80 mg/dl) = setiap 1 jam GD <200 mg/dL= Setiap 1 jam Stabil ...
```

Menggantikan `[PERKENI-2024-Pedoman-DMT2-Dewasa.pdf, N/A, Hal. N/A]` dari Versi 1.

**Retrieval kosong** (butir 7): `st.error` muncul **di atas** teks rekomendasi, bukan
sekadar menggantikan expander, sehingga jawaban tanpa rujukan tidak pernah disajikan
seolah-olah bersumber. `RAGPipeline.answer()` menambahkan flag `"grounded"` agar UI
tidak menurunkan sendiri status ini.

**Audit keputusan** (butir 8): `log_intervention` menyimpan `rag_sources` — **baris
tampilan hasil `build_source_list`**, bukan `retrieved_docs` mentah, karena rekaman
audit harus sama persis dengan yang dilihat dokter termasuk `page_label` yang sudah
diresolusi. Ditambah `rag_grounded`, `rag_query`, `rag_provider`, `rag_explanation`.
Dijaga tes `test_decision_log_persists_rag_sources_for_audit`.
```

**Kesimpulan untuk laporan:** sistem menyisipkan sitasi tingkat **dokumen** (nama berkas), **bukan tingkat halaman**. Klaim "sitasi halaman" tidak dapat didukung kode.

### G.7 Mekanisme fallback tiap lapisan

Lima lapisan fallback, dari ekstraksi sampai generasi.

**(1) Ekstraksi PDF gagal** — [scripts/reingest_kb.py:109-121](../scripts/reingest_kb.py#L109-L121):
```python
for p in pdfs:
    try:
        text = _clean_text(_read_pdf(p))
        if len(text) < args.min_chars:
            skipped.append((p.name, f"teks {len(text)} char — kemungkinan hasil scan/among gambar"))
            continue
        docs.append({...})
        added.append((p.name, len(text)))
    except Exception as exc:  # noqa: BLE001
        skipped.append((p.name, f"ERROR: {exc}"))
```
Dokumen gagal **dilewati**, sisanya tetap diproses. Jika **semua** dokumen gagal → `return 1` tanpa menyentuh Chroma ([:129-131](../scripts/reingest_kb.py#L129-L131)) — tetapi perhatikan bahwa penghapusan folder persist terjadi **setelah** cek ini ([:134-136](../scripts/reingest_kb.py#L134-L136)), jadi indeks lama tetap aman.

**(2) Splitter LangChain tidak tersedia** — [knowledge_base.py:206-210](../src/rag/knowledge_base.py#L206-L210):
```python
except Exception as exc:
    logger.warning("LangChain splitter unavailable, fallback splitter active: %s", exc)
    chunk_rows = self._fallback_chunk_documents(docs, chunk_size=chunk_size, overlap=chunk_overlap)
    self.chunks = chunk_rows
    return chunk_rows
```
Beralih ke chunker jendela-kata ([:212-246](../src/rag/knowledge_base.py#L212-L246)).

**(3) ChromaDB / embedding tidak tersedia saat retrieval** — dua tahap.

Konstruktor `MMRRetriever` menangkap kegagalan dan menandai dirinya tidak siap ([retriever.py:131-137](../src/rag/retriever.py#L131-L137)):
```python
except Exception as exc:
    self._init_error = str(exc)
    logger.warning(
        "MMRRetriever unavailable (embed=%s), keyword fallback required: %s",
        embed_provider, exc,
    )
```

`RAGPipeline.build()` lalu beralih ke retriever kata kunci — salinan persis ([pipeline.py:145-156](../src/rag/pipeline.py#L145-L156)):
```python
if mmr_retriever.is_ready:
    self.retriever = mmr_retriever
    logger.info("RAGPipeline: using MMR retriever (embed=%s)", self.embed_provider)
else:
    if not self.kb.chunks:
        docs = self.kb.load_manual_kb("manual_kb.json")
        if docs:
            self.kb.chunk_documents(documents=docs, chunk_size=350, chunk_overlap=40)
        else:
            self.kb.create_manual_kb()
    self.retriever = SimpleKeywordRetriever(self.kb.chunks)
    logger.info("RAGPipeline: using keyword retriever (Chroma unavailable)")
```
Penting: fallback ini hanya memuat **`manual_kb.json` (7 dokumen)** dengan chunk 350/40 — **tanpa PDF**. Jadi degradasi bukan sekadar "vektor → kata kunci", melainkan juga **korpus 2.585 chunk → ~30 chunk kurasi**.

**(4) Retrieval mengembalikan hasil kosong.** **Tidak ada penanganan khusus.** Tidak ada cek `if not retrieved_docs` di [pipeline.py:181-205](../src/rag/pipeline.py#L181-L205). Daftar kosong mengalir ke `format_context_with_citations()`, yang mengembalikan string penanda ([prompts.py:20-21](../src/rag/prompts.py#L20-L21)):
```python
if not retrieved_docs:
    return "(Tidak ada konteks dokumen yang ditemukan)"
```
LLM tetap dipanggil, dengan konteks kosong; aturan #3 system prompt yang diharapkan menanganinya. `advisory["source_count"]` akan bernilai 0 ([pipeline.py:315](../src/rag/pipeline.py#L315)).

**(5) API LLM error** — dua tingkat.

*Kegagalan inisialisasi* (API key tak ada, paket tak terpasang) — [advisor_chain.py:79-81](../src/rag/advisor_chain.py#L79-L81):
```python
except Exception as exc:
    self._init_error = str(exc)
    logger.warning("Advisor chain fallback active (provider=%s): %s", provider, exc)
```
`self._chain` tetap `None`.

*Kegagalan pemanggilan* (kuota habis, timeout, error jaringan) — salinan persis ([advisor_chain.py:97-115](../src/rag/advisor_chain.py#L97-L115)):
```python
if self._chain is not None:
    try:
        answer = self._chain.invoke(
            {
                "context": context_block,
                "question_payload": question_payload,
            }
        )
        return {
            "answer": answer.strip(),
            "sources": self._extract_sources(retrieved_docs),
        }
    except Exception as exc:
        logger.warning("Advisor chain invocation failed, using template: %s", exc)

return {
    "answer": self._template_answer(patient_state, prediction, retrieved_docs),
    "sources": self._extract_sources(retrieved_docs),
}
```

Jawaban template berbasis aturan ([advisor_chain.py:131-154](../src/rag/advisor_chain.py#L131-L154)) memakai ambang 70/180 dan menghasilkan teks tetap. **Sitasi tetap dikembalikan** meski jawaban berasal dari template. Template kedua yang serupa ada di [generator.py:85-109](../src/rag/generator.py#L85-L109) untuk `provider="template"`.

**(6) Lapisan pengaman keluaran.** Disclaimer dipaksa ada, apa pun sumber jawaban — salinan persis ([pipeline.py:338-342](../src/rag/pipeline.py#L338-L342)):
```python
def _ensure_disclaimer(self, text: str) -> str:
    disclaimer = "keputusan medis final tetap pada dokter"
    if disclaimer in text.lower():
        return text
    return f"{text.strip()} Catatan: Keputusan medis final tetap pada dokter."
```

Selain itu `advisory["actions"]` selalu diisi dari aturan deterministik `_actions_for_risk()` ([pipeline.py:319-336](../src/rag/pipeline.py#L319-L336)) — **tidak berasal dari LLM**, sehingga daftar tindakan tetap ada walau LLM gagal total.

**(7) Lapisan UI.** [app/streamlit_app.py:259-264](../app/streamlit_app.py#L259-L264):
```python
try:
    res = load_rag().answer(patient_state=patient_state, prediction=pred)
    st.session_state["last_rec"] = res
except Exception as exc:  # noqa: BLE001
    st.session_state["last_rec"] = None
    st.warning(f"Layanan rekomendasi (LLM) tidak tersedia: {str(exc)[:90]}")
```
Prediksi dan peringatan klinis tetap tampil; hanya blok rekomendasi yang hilang.

**Ringkasan degradasi bertingkat:**
```
Gemini 2.5 Flash-Lite    →  template berbasis aturan (advisor_chain)
MMR vektor (2.061 chunk) →  keyword overlap (~30 chunk manual_kb)
PDF gagal / tak terdaftar →  ABORT (bukan lagi dilewati diam-diam)
LangChain splitter       →  chunker jendela-kata
skor kemiripan gagal     →  MMR tanpa skor (similarity = None)
retrieval kosong         →  st.error di UI; jawaban TIDAK disajikan sebagai bersumber
apa pun yang terjadi     →  disclaimer + actions deterministik tetap ada
```

> **BERUBAH (Tugas 1 & 1B):** dua baris berubah maknanya.
> **(a)** PDF yang gagal atau tidak terdaftar di manifest kini **menghentikan proses**,
> bukan dilewati diam-diam — mengindeks 11 dari 12 dokumen tanpa peringatan sama
> berbahayanya dengan mengindeks dokumen yang salah.
> **(b)** Retrieval kosong kini ditangani eksplisit di UI; Versi 1 mencatat "tanpa
> penanganan khusus; LLM tetap dipanggil".

---

## H. STRUKTUR PERANGKAT LUNAK

### H.1 Pohon direktori `src/` sampai kedalaman 3

> **BERUBAH (Tugas 2, 3, 4):** paket `src/digital_twin/` **dihapus** dan digantikan
> `src/clinical_state/`; dua modul akar baru: `src/constants.py` (Tugas 3) dan
> `src/config.py` (Tugas 4); satu modul RAG baru: `src/rag/citations.py` (Tugas 1B).

```
src/
├── __init__.py                    (kosong)
├── constants.py                   BARU (Tugas 3) — ambang glikemik + 2 fungsi klasifikasi
├── config.py                      BARU (Tugas 4) — load_config, RagConfig, resolve()
├── patient_state.py               PatientState — kontrak data ML → RAG
├── clinical_state/                BARU (Tugas 2) — menggantikan digital_twin/
│   ├── __init__.py                ekspor: ClinicalDecisionLog, StateRecord
│   └── decision_log.py            ClinicalDecisionLog (dulu DigitalTwinStateManager)
├── data/
│   ├── __init__.py                ekspor: DiabetesDataLoader, DatasetInfo, DataPreprocessor
│   ├── contracts.py               validate_data_contract, assert_feature_set
│   ├── loader.py                  DiabetesDataLoader, DatasetInfo
│   ├── ohio_parser.py             parser XML OhioT1DM → CSV
│   └── preprocessor.py            DataPreprocessor
├── models/
│   ├── __init__.py                ekspor: RandomForestGlucoseModel
│   ├── base_model.py              BaseGlucoseModel (ABC)
│   ├── lstm_model.py              LSTMGlucoseModel + train_lstm_from_config
│   └── rf_model.py                RandomForestGlucoseModel + train_random_forest_from_config
├── rag/
│   ├── __init__.py                ekspor: 11 simbol
│   ├── advisor_chain.py           DiabetesAdvisorChain
│   ├── citations.py               BARU (Tugas 1B) — format_page_label, build_source_list
│   ├── conditioned_query.py       PredictionConditionedQueryBuilder, ConditionedQuery, QueryStrategy
│   ├── generator.py               RAGGenerator
│   ├── knowledge_base.py          MedicalKnowledgeBase
│   ├── pipeline.py                RAGPipeline, RetrievedDocument
│   ├── prompts.py                 SYSTEM_PROMPT, format_context_with_citations, build_question_payload
│   └── retriever.py               MMRRetriever, SimpleKeywordRetriever, DocumentRetriever (alias)
└── utils/
    ├── __init__.py                (kosong)
    ├── logger.py                  setup_logger
    └── metrics.py                 rmse, mae, mape, clarke_error_grid, calculate_all_metrics
```

**Yang dihapus (Tugas 2):**
```
src/digital_twin/patient_twin.py   429 baris — PatientDigitalTwin, TwinManager, mesin PK
src/digital_twin/simulator.py      206 baris — WhatIfSimulator
```

Kedalaman maksimum sebenarnya adalah 2 (`src/<paket>/<modul>.py`); tidak ada sub-paket bersarang.

**Verifikasi penghapusan:**
```
$ grep -rniE "digital_twin|DigitalTwin|what_?if|WhatIf" src/ app/ --include=*.py
$ echo $?
1        # NOL kecocokan
```
Sisa kata "twin" di seluruh `src/` dan `app/` hanya **satu**: judul paper Rad et al.
(2024) yang disitasi pada docstring `conditioned_query.py` — rujukan pustaka ke karya
orang lain, sengaja dipertahankan.

Direktori tingkat atas lain:
```
app/                    Streamlit UI
├── streamlit_app.py    halaman utama (Konsultasi)
├── ui.py               komponen UI bersama
└── pages/
    ├── 1_Input_Logbook.py
    └── 2_Tentang_dan_Validasi.py
scripts/                20 skrip evaluasi/pelatihan/ingest
tests/                  12 berkas pytest
models/                 artefak terlatih + chroma_db
data/                   raw, knowledge_base, processed
docs/                   dokumentasi
run_app.py              launcher (import torch sebelum Streamlit)
config.yaml
```

### H.2 Kelas utama dan method publiknya

**`src/data/`**

- **`DiabetesDataLoader`** ([loader.py:23](../src/data/loader.py#L23))
  `load_csv`, `load_latest_dataset`, `load_preferred_dataset`, `list_datasets`, `load_patient_data`
- **`DatasetInfo`** ([loader.py:15](../src/data/loader.py#L15)) — dataclass: `path`, `rows`, `columns`
- **`DataPreprocessor`** ([preprocessor.py:31](../src/data/preprocessor.py#L31))
  `handle_missing_values`, `engineer_features`, `create_sequences`, `normalize_data`, `downsample_smbg`, `split_by_patient`
- Fungsi modul `contracts.py`: `validate_data_contract`, `assert_feature_set`
- Fungsi modul `ohio_parser.py`: `parse_ohio_xml`, `parse_ohio_fingerstick`, `process_ohio_dataset`

**`src/models/`**

- **`BaseGlucoseModel`** (ABC, [base_model.py:10](../src/models/base_model.py#L10))
  `train` (abstract), `predict` (abstract), `save` (abstract), `load` (abstract), `evaluate`
- **`RandomForestGlucoseModel`** ([rf_model.py:21](../src/models/rf_model.py#L21))
  `train`, `predict`, `save`, `load`
- **`LSTMGlucoseModel`** ([lstm_model.py:24](../src/models/lstm_model.py#L24))
  `train`, `predict`, `save`, `load`
- Fungsi modul: `train_random_forest_from_config`, `train_lstm_from_config`

**`src/patient_state.py`**

- **`PatientState`** (dataclass, [:35](../src/patient_state.py#L35))
  `to_dict`, `to_rag_context`, `from_model_output` (classmethod), `from_digital_twin` (classmethod), `from_dict` (classmethod)

**`src/digital_twin/`**

- **`PatientDigitalTwin`** ([patient_twin.py:15](../src/digital_twin/patient_twin.py#L15))
  `update_state`, `calculate_insulin_on_board`, `calculate_carbs_on_board`, `predict_glucose_impact`, `simulate_scenario`, `get_state_summary`, `export_state`, `load_from_json` (classmethod)
- **`TwinManager`** ([patient_twin.py:356](../src/digital_twin/patient_twin.py#L356))
  `create_twin`, `get_twin`, `save_all_twins`, `load_all_twins`
- **`WhatIfSimulator`** ([simulator.py:15](../src/digital_twin/simulator.py#L15))
  `compare_meal_scenarios`, `simulate_stress_reduction`, `simulate_exercise`, `find_optimal_insulin_dose`, `generate_decision_tree`
- **`DigitalTwinStateManager`** ([state_manager.py:78](../src/digital_twin/state_manager.py#L78))
  `create_state`, `get_state`, `update_state`, `log_intervention`, `get_events`, `append_event`, `list_patients`, `save`, `load`
- **`StateRecord`** ([state_manager.py:70](../src/digital_twin/state_manager.py#L70)) — dataclass

**`src/rag/`**

- **`RAGPipeline`** ([pipeline.py:37](../src/rag/pipeline.py#L37))
  `ingest`, `build`, `answer`
- **`RetrievedDocument`** ([pipeline.py:21](../src/rag/pipeline.py#L21)) — dataclass: `rank`, `text`, `source`, `similarity`, `metadata`
- **`MedicalKnowledgeBase`** ([knowledge_base.py:68](../src/rag/knowledge_base.py#L68))
  `load_manual_kb`, `load_documents`, `chunk_documents`, `save_to_chroma`, `save_chunks`, `load_chunks`, `process_all_documents`, `create_manual_kb`
- **`MMRRetriever`** ([retriever.py:91](../src/rag/retriever.py#L91)) — alias `DocumentRetriever`
  `is_ready` (property), `retrieve`, `retrieve_with_context`
- **`SimpleKeywordRetriever`** ([retriever.py:51](../src/rag/retriever.py#L51))
  `retrieve`
- **`RAGGenerator`** ([generator.py:11](../src/rag/generator.py#L11))
  `generate_explanation`, `generate_advisory`
- **`DiabetesAdvisorChain`** ([advisor_chain.py:14](../src/rag/advisor_chain.py#L14))
  `is_ready` (property), `generate`
- **`PredictionConditionedQueryBuilder`** ([conditioned_query.py:64](../src/rag/conditioned_query.py#L64))
  `build`
- **`ConditionedQuery`** ([conditioned_query.py:246](../src/rag/conditioned_query.py#L246))
  `to_pipeline_kwargs`
- **`QueryStrategy`** ([conditioned_query.py:52](../src/rag/conditioned_query.py#L52)) — `str, Enum`: `RISK_FOCUSED`, `INTERVENTION_FOCUSED`, `MONITORING_FOCUSED`, `COMPREHENSIVE`
- Fungsi modul: `build_conditioned_query`, `format_context_with_citations`, `build_question_payload`

**`src/utils/`**
- `setup_logger` ([logger.py:10](../src/utils/logger.py#L10))
- `rmse`, `mae`, `mape`, `clarke_error_grid`, `calculate_all_metrics` ([metrics.py](../src/utils/metrics.py))

### H.3 Ketergantungan antarmodul

Berdasarkan pernyataan `import` aktual:

```
app/streamlit_app.py
├── src.data.loader          → DiabetesDataLoader
├── src.digital_twin         → DigitalTwinStateManager, PatientDigitalTwin, WhatIfSimulator
├── src.rag                  → RAGPipeline
├── src.data.preprocessor    → DataPreprocessor   (impor tertunda, streamlit_app.py:94)
└── app/ui.py                → komponen UI

app/pages/1_Input_Logbook.py       → app/ui.py saja
app/pages/2_Tentang_dan_Validasi.py → app/ui.py saja

src/models/rf_model.py
├── src.data.loader          → DiabetesDataLoader
├── src.data.preprocessor    → DataPreprocessor
├── src.models.base_model    → BaseGlucoseModel
└── src.utils.metrics        → calculate_all_metrics

src/models/lstm_model.py
├── src.models.base_model    → BaseGlucoseModel
├── src.utils.metrics        → calculate_all_metrics
└── (src.data.loader, src.data.preprocessor — impor tertunda di dalam fungsi, lstm_model.py:156-157)

src/data/loader.py           → src.data.contracts
src/data/preprocessor.py     → src.data.contracts

src/rag/pipeline.py
├── .conditioned_query       → PredictionConditionedQueryBuilder, QueryStrategy
├── .generator               → RAGGenerator
├── .knowledge_base          → MedicalKnowledgeBase
├── .retriever               → MMRRetriever, SimpleKeywordRetriever
└── src.patient_state        → PatientState   (impor tertunda + fallback relatif, pipeline.py:248-250)

src/rag/generator.py         → .advisor_chain → DiabetesAdvisorChain
src/rag/advisor_chain.py     → .prompts       → SYSTEM_PROMPT, build_question_payload,
                                                format_context_with_citations
src/rag/conditioned_query.py → src.patient_state (dengan fallback relatif, :42-45)

src/digital_twin/simulator.py → .patient_twin → PatientDigitalTwin
```

Diagram lapisan:
```
      app/streamlit_app.py
       │        │        │
       ▼        ▼        ▼
  src.data  src.rag  src.digital_twin
       │        │        │
       │        ▼        ▼
       │  src.patient_state   (kontrak ML → RAG)
       ▼
  src.data.contracts   (kontrak data kanonik)

  src.models ──► src.data ──► src.data.contracts
       └───────► src.utils.metrics
```

Pengamatan penting:
- **`src.models` tidak diimpor oleh aplikasi sama sekali.** Aplikasi memuat `models/rf_inference_bundle.pkl` langsung dengan `pickle.load` ([streamlit_app.py:68](../app/streamlit_app.py#L68)) dan memanggil `art["model"].predict()` pada objek scikit-learn mentah — kelas `RandomForestGlucoseModel` di-bypass saat inferensi.
- **`src.rag` tidak bergantung pada `src.models` maupun `src.data`.** Kopling terjadi lewat `src.patient_state` dan lewat dict biasa.
- **Tidak ada dependensi melingkar.** `src.patient_state` sengaja diletakkan di akar `src/`, bukan di dalam `src/rag/`, agar `rag` dan `digital_twin` sama-sama bisa mengimpornya.
- Beberapa impor sengaja **tertunda** (di dalam fungsi) untuk menghindari biaya startup TensorFlow/torch: [lstm_model.py:42](../src/models/lstm_model.py#L42), [lstm_model.py:65](../src/models/lstm_model.py#L65), [knowledge_base.py:174](../src/rag/knowledge_base.py#L174), [retriever.py:118](../src/rag/retriever.py#L118), [streamlit_app.py:94](../app/streamlit_app.py#L94).

### H.4 Halaman Streamlit

Tiga halaman (multipage Streamlit standar: berkas utama + direktori `pages/`).

**(1) `app/streamlit_app.py` — "Konsultasi"** (halaman utama, ikon 🩺)

Fungsi: konsol utama klinisi. Alur, sesuai [streamlit_app.py:152](../app/streamlit_app.py#L152): *tinjau status → prediksi & risiko → rekomendasi → simulasi/keputusan*.

| Bagian | Baris | Isi |
|--------|-------|-----|
| Sidebar | [143-152](../app/streamlit_app.py#L143-L152) | Pemilih pasien, tampilan horizon |
| Section 1 | [174-204](../app/streamlit_app.py#L174-L204) | Chart zona glukosa + titik prediksi, badge risiko, metrik glukosa sekarang/prediksi/tren, rentang keyakinan conformal, kartu "Kondisi aktif" |
| Peringatan divergen | [206-209](../app/streamlit_app.py#L206-L209) | Current normal → prediksi bahaya |
| Peringatan hipo dini | [211-229](../app/streamlit_app.py#L211-L229) | Pengklasifikasi kondisi + batas bawah interval |
| Tab "🧠 Rekomendasi Klinis" | [236-276](../app/streamlit_app.py#L236-L276) | Memanggil `RAGPipeline.answer()`, menampilkan penjelasan, daftar tindakan, expander rujukan |
| Tab "🔬 Simulasi What-If" | [278-299](../app/streamlit_app.py#L278-L299) | Slider karbohidrat/insulin/aktivitas/horizon → `twin.simulate_scenario()` |
| Tab "📝 Catat Keputusan" | [301-318](../app/streamlit_app.py#L301-L318) | `DigitalTwinStateManager.log_intervention()` ke `data/processed/patient_states.json` |

Fungsi lokal ber-cache: `load_dataset` ([31](../app/streamlit_app.py#L31)), `load_condition_classifier` ([37](../app/streamlit_app.py#L37)), `load_artifacts` ([63](../app/streamlit_app.py#L63)), `load_rag` ([78](../app/streamlit_app.py#L78)); helper `build_window` ([90](../app/streamlit_app.py#L90)), `predict_next` ([101](../app/streamlit_app.py#L101)), `predict_uncertainty` ([112](../app/streamlit_app.py#L112)), `predict_condition` ([51](../app/streamlit_app.py#L51)).

**(2) `app/pages/1_Input_Logbook.py` — "Input Logbook"** (ikon 📝, 103 baris)

Fungsi: entri data logbook harian secara manual. Form berisi ID pasien, tanggal/waktu, glukosa, karbohidrat, insulin, aktivitas, stres (slider 1–10), checkbox tidur/kerja/sakit, jenis makan, catatan. Menulis-tambah ke `data/raw/manual_logbook.csv` ([:16](../app/pages/1_Input_Logbook.py#L16)) dengan 13 kolom ([:17-18](../app/pages/1_Input_Logbook.py#L17-L18)). Menampilkan riwayat dan chart tren.

Perlu dicatat: berkas ini adalah **jalur masuk data logbook manual**, tetapi `manual_logbook.csv` hanya bisa dimuat lewat `_load_by_source("manual_logbook")` ([loader.py:83-84](../src/data/loader.py#L83-L84)) — dan halaman Konsultasi memuat `"ohio_t1dm"` ([streamlit_app.py:33](../app/streamlit_app.py#L33)). Jadi data yang dicatat di sini **tidak mengalir ke halaman prediksi**.

**(3) `app/pages/2_Tentang_dan_Validasi.py` — "Tentang & Validasi"** (ikon ℹ️, 79 baris)

Fungsi: transparansi untuk klinisi. Menjelaskan sistem sebagai decision support doctor-mediated dan status prototipe riset TA. Membaca `results/eval_prediksi/summary_all_horizons.csv` (atau fallback `models/rf_baseline_metrics.json`) dan menyajikan RMSE + Clarke A+B dalam bahasa klinis, bukan bahasa metrik ([:35-58](../app/pages/2_Tentang_dan_Validasi.py#L35-L58)). Berisi bagian "Cara Kerja Singkat" dan "Keterbatasan".

**Komponen bersama** — [app/ui.py](../app/ui.py): `classify_glucose`, `inject_global_css`, `app_header`, `risk_badge`, `glucose_zone_chart`, `disclaimer_footer`, `zone_legend`. Ambang klinis UI di [ui.py:14-15](../app/ui.py#L14-L15) (70/180), palet warna di [ui.py:17-21](../app/ui.py#L17-L21).

**Launcher:** [run_app.py](../run_app.py) — wajib dipakai di Windows karena `torch` harus di-import sebelum Streamlit menyentuh numpy/pyarrow (WinError 1114 pada `c10.dll`).

### H.5 Konfirmasi: kode what-if simulation dan PK engine

**Ya, keduanya masih ada. Dan keduanya AKTIF DIPAKAI di aplikasi — bukan kode mati.**

**Path lengkap:**

| Path | Isi | Status |
|------|-----|--------|
| [src/digital_twin/patient_twin.py](../src/digital_twin/patient_twin.py) | `PatientDigitalTwin` (429 baris) — berisi mesin PK dan `simulate_scenario` | **AKTIF** |
| [src/digital_twin/simulator.py](../src/digital_twin/simulator.py) | `WhatIfSimulator` — skenario tingkat lebih tinggi | **DIIMPOR & DIINSTANSIASI, TAPI HASILNYA TIDAK DIPAKAI** |
| [src/digital_twin/state_manager.py](../src/digital_twin/state_manager.py) | `DigitalTwinStateManager` | **AKTIF** (tab Catat Keputusan) |
| [src/digital_twin/__init__.py](../src/digital_twin/__init__.py) | Ekspor keempat kelas | **AKTIF** |
| [tests/test_digital_twin.py](../tests/test_digital_twin.py), [tests/test_digital_twin_stability.py](../tests/test_digital_twin_stability.py), [tests/test_simulation_consistency.py](../tests/test_simulation_consistency.py), [tests/test_state_manager.py](../tests/test_state_manager.py) | Uji | ada |

**Komponen PK (mekanistik) di dalam `patient_twin.py`:**

- Parameter fisiologis hardcoded ([:52-59](../src/digital_twin/patient_twin.py#L52-L59)): `insulin_sensitivity=50.0` mg/dL per unit, `carb_ratio=10.0` g per unit, `insulin_duration_hours=4.0`, `carb_absorption_hours=3.0`, `stress_glucose_impact=2.0`, `activity_glucose_impact=-1.5`.
- `calculate_insulin_on_board()` ([:134-154](../src/digital_twin/patient_twin.py#L134-L154)) — peluruhan eksponensial berbasis paruh waktu: `dose * exp(-decay_rate * hours)`, `decay_rate = ln(2)/(duration/2)`.
- `calculate_carbs_on_board()` ([:156-177](../src/digital_twin/patient_twin.py#L156-L177)) — penyerapan **linear**.
- `predict_glucose_impact()` ([:179-209](../src/digital_twin/patient_twin.py#L179-L209)) — model efek aditif, diskalakan linear terhadap horizon, di-clip ke [40, 400].

Perhatikan: dua fungsi PK pertama (`calculate_insulin_on_board`, `calculate_carbs_on_board`) **tidak dipanggil dari mana pun dalam kode produksi** — hanya dari tes. `predict_glucose_impact()` yang benar-benar dipakai, dan ia membaca `state['insulin_on_board']`/`state['carbs_on_board']` apa adanya.

**Jalur aktif di aplikasi** — tab "🔬 Simulasi What-If" ([streamlit_app.py:278-299](../app/streamlit_app.py#L278-L299)):
```python
twin = PatientDigitalTwin(patient_id=sel, initial_state={...})   # baris 281
sim = WhatIfSimulator(twin)                                       # baris 286
...
r = twin.simulate_scenario({"carbs_delta": …, "insulin_delta": …,
                            "activity_delta": …, "stress_delta": 0,
                            "time_horizon": int(hz)})             # baris 293
```

**Satu temuan yang relevan untuk keputusan Anda:** variabel `sim` dibuat di baris 286 tetapi **tidak pernah dipakai**. Tombol simulasi memanggil `twin.simulate_scenario()` secara langsung. Jadi seluruh kelas `WhatIfSimulator` (`compare_meal_scenarios`, `simulate_stress_reduction`, `simulate_exercise`, `find_optimal_insulin_dose`, `generate_decision_tree`) **tidak pernah dieksekusi di jalur aplikasi** — hanya di tes.

**Ringkasan untuk keputusan hapus/biarkan:**

| Komponen | Dipakai aplikasi? | Rekomendasi faktual |
|----------|-------------------|---------------------|
| `PatientDigitalTwin.simulate_scenario` + `predict_glucose_impact` | **Ya**, tab What-If | Menghapusnya akan merusak tab Simulasi |
| `PatientDigitalTwin.calculate_insulin_on_board` / `calculate_carbs_on_board` | Tidak (hanya tes) | Kode mati secara efektif |
| `WhatIfSimulator` (seluruh kelas) | Tidak (diinstansiasi tetapi tak terpakai) | Kode mati secara efektif; baris 286 bisa dihapus tanpa efek |
| `DigitalTwinStateManager` | **Ya**, tab Catat Keputusan | Aktif |
| `TwinManager` | Tidak | Tidak pernah diimpor aplikasi |

Perlu juga dicatat untuk konsistensi laporan: mesin PK ini **terpisah sepenuhnya** dari IOB/COB yang dipakai model prediksi. `DataPreprocessor._decay_accumulate()` (B.1) memakai akumulator IIR dengan τ dari `config.yaml`; `PatientDigitalTwin` memakai peluruhan berbasis paruh waktu dengan konstanta hardcoded. Keduanya **tidak berbagi kode maupun nilai**, dan tab What-If tidak memakai model Random Forest sama sekali. Ini disebut eksplisit di UI ([streamlit_app.py:279-280](../app/streamlit_app.py#L279-L280)): *"Memakai model farmakokinetik mekanistik agar arah kausal (insulin↓, karbohidrat↑) benar."*

---

## I. KONFIGURASI

Isi `config.yaml` apa adanya:

```yaml
# Configuration for Diabetes Digital Twin System

# Data Generation Settings
data:
  num_patients: 10
  days_per_patient: 7
  sampling_interval_min: 5
  output_dir: "data/raw"
  seed: 42
  primary_source: "ohio_t1dm"
  fallback_source: "latest_generated"
  # Skenario SMBG memakai pembacaan finger_stick NYATA dari OhioT1DM
  # (data/raw/ohio_t1dm_smbg.csv, dihasilkan parse_ohio_fingerstick) — BUKAN hasil
  # downsampling artifisial dari kanal CGM. Lihat Batasan Masalah #2 pada laporan.
  smbg_csv: "data/raw/ohio_t1dm_smbg.csv"
  # Dipakai HANYA oleh jalur eksperimen lama train_lstm_from_config(smbg_downsample=True),
  # yang tidak dipakai pada hasil mana pun di laporan. Dipertahankan untuk reproduksi historis.
  smbg_interval_min: 240

# [BLOK patient: DAN digital_twin: DIHAPUS pada Tugas 2 — keduanya tidak pernah
#  dibaca kode; konstanta patient: terduplikasi hardcoded di patient_twin.py yang
#  ikut terhapus.]

# Model Training
model:
  name: "RandomForest" # Options: RandomForest, Simulation
  sequence_length: 12 # look-back 1 jam (12 x 5 menit)
  # Horizon prediksi (langkah ke depan, cadence 5-menit) — standar BGLP Challenge:
  #   6  = +30 menit, 12 = +60 menit
  prediction_horizons: [6, 12]
  default_horizon: 6   # horizon untuk bundle inferensi aplikasi (30 menit)
  # stress dikeluarkan: OhioT1DM nyaris tak punya data stres (7 event di seluruh
  # dataset) sehingga nol-varians & tak informatif bagi model. Tetap dipertahankan
  # sebagai variabel Digital Twin / logbook (lihat docs/journey.md).
  features: ["glucose", "carbs", "insulin", "activity"]

  # Fitur hasil rekayasa (engineered) berbasis fisiologi — dipakai bila use_engineered=true.
  # iob/cob menangkap efek insulin/karbohidrat yang tertunda; glucose_delta = tren;
  # hour_sin/cos = pola diurnal. glucose tetap fitur pertama (index 0 = anchor).
  use_engineered: true
  engineered_features: ["glucose", "glucose_delta", "iob", "cob", "activity", "hour_sin", "hour_cos"]
  predict_delta: true   # model memprediksi Δglukosa (selisih dari nilai terakhir), lalu direkonstruksi

  # [BARU Tugas 5] Segmentasi jendela pada jeda sensor. Jendela dibentuk per posisi
  # baris, padahal baris OhioT1DM tidak berjarak seragam: jeda terpanjang 118 jam.
  # Jendela yang memuat jeda antar-baris > max_gap_steps langkah DIBUANG, bukan
  # diinterpolasi. 6 langkah x 5 menit = 30 menit.
  max_gap_steps: 6
  # Batas panjang runtun NaN yang masih boleh diinterpolasi. Tidak berpengaruh pada
  # ohio_t1dm_merged.csv (nol NaN); pengaman untuk manual_logbook.csv.
  max_interpolate_steps: 6

  feature_engineering:
    insulin_tau_min: 240   # waktu kerja insulin ~4 jam
    carbs_tau_min: 180     # penyerapan karbohidrat ~3 jam
    trend_steps: 3         # tren glukosa atas 3 langkah (15 menit)

  # sequence_length=12 → look-back 1 jam pada cadence CGM 5-menit
  # smbg_sequence_length=6 → look-back ~1 hari pada cadence SMBG 240-menit
  smbg_sequence_length: 6

  random_forest:
    n_estimators: 200
    max_depth: 20
    min_samples_split: 5

  lstm:
    units_1: 64
    units_2: 32
    dropout: 0.2
    learning_rate: 0.001
    epochs: 100
    patience: 10

# RAG Settings
# Seluruh nilai di blok ini BENAR-BENAR DIBACA kode lewat src/config.py (Tugas 4).
# Presedensi: argumen konstruktor > variabel environment > config.yaml > default kode.
rag:
  knowledge_base_dir: "data/knowledge_base"
  persist_dir: "models/chroma_db"
  collection_name: "diabetes_kb"

  # Chunking dokumen pedoman (per halaman PDF)
  chunk_size: 900
  chunk_overlap: 120
  # Entri manual_kb adalah prosa pendek, bukan halaman buku — sengaja lebih kecil.
  manual_kb:
    chunk_size: 350
    chunk_overlap: 40

  # Embedding
  embedding_provider: "sentence-transformers"  # sentence-transformers | google | ollama
  embedding_model: "all-MiniLM-L6-v2"          # 384 dimensi, CPU
  google_embedding_model: "models/embedding-001"

  # Retrieval (MMR)
  top_k_retrieval: 4    # jumlah dokumen yang dikembalikan
  fetch_k: 12           # ukuran kolam kandidat sebelum diversifikasi MMR
  lambda_mult: 0.5      # 1,0 = relevansi murni; 0,0 = keragaman maksimum

  ollama:
    base_url: "http://localhost:11434"
    embed_model: "nomic-embed-text"

  llm:
    provider: "gemini"  # Options: gemini, ollama, template
    model: "gemini-2.5-flash-lite"  # Kuota free-tier gemini-2.5-flash-lite (diverifikasi 2026-08):
                                    # 15 request/menit, 1.000 request/HARI. Cukup untuk aplikasi
                                    # maupun RAGAS (~110 panggilan untuk 10 kasus x 4 metrik).
                                    # flash-lite dipilih karena non-thinking (latensi rendah).
    ollama_model: "llama3.1:8b"
    # Temperature rendah demi keselamatan klinis: menekan risiko halusinasi pada
    # nasihat medis. Nilai >0,5 menghasilkan bahasa lebih bervariasi tetapi berisiko
    # mengarang angka dosis/ambang.
    temperature: 0.2
    # Batas panjang keluaran. Sebelum Tugas 4 nilai ini tidak pernah diterapkan;
    # 300 token terlalu ketat untuk nasihat Bahasa Indonesia beserta disclaimer.
    max_tokens: 700

# Evaluation
evaluation:
  test_split: 0.2
  metrics: ["RMSE", "MAE", "MAPE", "ClarkeErrorGrid"]
  clarke_zones: ["A", "B", "C", "D", "E"]

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/system.log"
```

### I.1 Kunci mana yang benar-benar dibaca kode

> **BERUBAH TOTAL (Tugas 4):** Versi 1 mencatat *"sebagian besar blok `rag:` tidak
> pernah dibaca kode"*. Setelah Tugas 4, **tidak ada lagi parameter di `config.yaml`
> yang diabaikan kode.** Yang tidak relevan sudah dihapus (Tugas 2), sisanya dibaca.

**Aturan presedensi, berlaku di mana pun:**

> argumen konstruktor eksplisit > variabel environment > `config.yaml` > default kode

Disertai kebijakan **tidak membuat variabel environment baru untuk tombol perilaku**
(`chunk_size`, `chunk_overlap`, `top_k`, `fetch_k`, `lambda_mult`, `temperature`,
`max_tokens`, `embedding_model`). Karena env semacam itu tidak ada, `config.yaml`
menjadi otoritatif *de facto* untuk parameter tersebut, sementara env tetap otoritatif
untuk pengaturan per-mesin dan rahasia (`GOOGLE_API_KEY`, `LLM_PROVIDER`,
`EMBED_PROVIDER`, `GEMINI_MODEL`).

Bila env dan config berbeda untuk parameter yang sama, `resolve()` menulis peringatan
ke log — tanpa itu laporan bisa mendokumentasikan model yang tidak dijalankan.

**Dibaca (jalur produksi):**

| Kunci | Pembaca |
|-------|---------|
| `data.output_dir` | [rf_model.py:103](../src/models/rf_model.py#L103), [lstm_model.py:162](../src/models/lstm_model.py#L162), [conformal_calibration.py:50](../scripts/conformal_calibration.py#L50) |
| `data.seed` | [rf_model.py:32](../src/models/rf_model.py#L32) → `random_state` |
| `data.primary_source`, `data.fallback_source` | [rf_model.py:105-106](../src/models/rf_model.py#L105-L106) |
| `data.smbg_interval_min` | [lstm_model.py:181](../src/models/lstm_model.py#L181) (jalur legacy) |
| `model.sequence_length` | [rf_model.py:120](../src/models/rf_model.py#L120) |
| `model.default_horizon`, `model.prediction_horizons` | [rf_model.py:121](../src/models/rf_model.py#L121), [rf_model.py:224](../src/models/rf_model.py#L224) |
| `model.use_engineered`, `model.predict_delta` | [rf_model.py:126-127](../src/models/rf_model.py#L126-L127) |
| `model.engineered_features` / `model.features` | [rf_model.py:131-133](../src/models/rf_model.py#L131-L133) |
| `model.feature_engineering.*` | [rf_model.py:128](../src/models/rf_model.py#L128) |
| `model.random_forest.*` | [rf_model.py:26-31](../src/models/rf_model.py#L26-L31) |
| `model.lstm.*` | [lstm_model.py:29-37](../src/models/lstm_model.py#L29-L37) |
| `model.smbg_sequence_length` | [lstm_model.py:184](../src/models/lstm_model.py#L184) |

**Kini juga dibaca (seluruhnya lewat `RagConfig`, Tugas 4):**

| Kunci | Versi 1 | **Nilai efektif Versi 2** |
|-------|---------|---------------------------|
| `rag.llm.temperature` | 0.1 hardcoded | **0.2** dari config |
| `rag.llm.max_tokens` | tidak diterapkan | **700** dari config |
| `rag.chunk_size` / `chunk_overlap` | konstanta modul | **900 / 120** dari config |
| `rag.manual_kb.chunk_size` / `overlap` | 350/40 hardcoded | **350 / 40** dari config |
| `rag.top_k_retrieval` | default ctor | **4** dari config |
| `rag.fetch_k` | `max(top_k*3,12)` | **12** dari config |
| `rag.lambda_mult` | 0.5 hardcoded | **0.5** dari config |
| `rag.embedding_model` | konstanta modul ×2 | **all-MiniLM-L6-v2** dari config |
| `rag.google_embedding_model` | hardcoded ×2 | **models/embedding-001** dari config |
| `rag.persist_dir` / `collection_name` | hardcoded ×3 | dari config (env boleh menimpa) |
| `rag.ollama.base_url` / `embed_model` | hardcoded ×4 | dari config |
| `rag.llm.ollama_model` | `"llama3.1:8b"` | dari config |
| `model.max_gap_steps` | tidak ada | **6** (Tugas 5) |
| `model.max_interpolate_steps` | tidak ada | **6** (Tugas 5) |

**Masih tidak dibaca** (sisa yang sengaja dibiarkan):

| Kunci | Alasan |
|-------|--------|
| `data.num_patients: 10` | 12 pasien aktual; tidak dipakai parser |
| `data.days_per_patient` | tidak dipakai |
| `model.name: "RandomForest"` | tidak dipakai |
| `evaluation.*` | tidak dipakai; metrik dihitung `src/utils/metrics.py` |
| `logging.*` | `logger.py` memakai parameter sendiri |

Blok `patient:` dan `digital_twin:` yang dulu ada di daftar ini **sudah dihapus dari
config** pada Tugas 2, sehingga tidak lagi menjadi parameter yang diabaikan.

`data.sampling_interval_min` yang dulu tidak dipakai **kini dibaca** oleh Tugas 5
sebagai cadence untuk mengonversi `max_gap_steps` ke menit.

Berkas `.env` (kredensial disamarkan):
```
LLM_PROVIDER=gemini
GOOGLE_API_KEY=<redacted>
GEMINI_MODEL=gemini-2.5-flash-lite
EMBED_PROVIDER=sentence-transformers
CHROMA_PERSIST_DIR=models/chroma_db
CHROMA_COLLECTION_NAME=diabetes_kb
```

Catatan: `CHROMA_PERSIST_DIR` dan `CHROMA_COLLECTION_NAME` ada di `.env` tetapi **tidak dibaca kode mana pun** — nilai yang sama datang dari default konstruktor [pipeline.py:58-59](../src/rag/pipeline.py#L58-L59).

---

## Lampiran: Daftar temuan — status setelah refaktor

Hal-hal berikut ditemukan saat pembacaan kode Versi 1. Kolom **Status** menunjukkan
apakah temuan itu sudah ditangani oleh refaktor tujuh tugas.

**Ringkasan: 16 dari 22 temuan Versi 1 selesai, 6 masih terbuka.**

| # | Temuan | Bagian | Status |
|---|--------|--------|--------|
| 1 | `METHODOLOGY.md:147` menyebut "8 dari 10 pasien". Kode: **12 pasien, 10 latih / 2 uji**. | A.6 | **TERBUKA** — perlu koreksi di METHODOLOGY.md |
| 2 | `temp_basal` tidak pernah dibaca; periode suspensi pompa dihitung memakai rate basal terjadwal. | A.4 | **TERBUKA** — di luar cakupan tujuh tugas |
| 3 | Kolom `sleep` dan `work` selalu 0 — kode meminta atribut `ts`, XML memakai `ts_begin`. | A.2 | **TERBUKA** — di luar cakupan |
| 4 | Interpolasi nilai kosong tanpa batas panjang gap. | A.5 | **SELESAI (Tugas 5)** — `max_interpolate_steps`; lihat catatan di bawah |
| 5 | Kolom `activity` berisi skor intensitas, bukan menit, tetapi UI menyebutnya menit. | B.5 | **SELESAI (perbaikan lanjutan)** — konsisten disebut "skor intensitas" |
| 6 | IOB dihitung dari `bolus_dose` saja; rumusnya akumulator IIR orde-1, bukan kurva PK. | B.1 | **TERBUKA** — pilihan desain, bukan cacat |
| 7 | LSTM `train_lstm_from_config()` tidak sebanding langsung dengan RF. | C.2 | **TERBUKA** — di luar cakupan |
| 8 | Faktor conformal di aplikasi konstanta hardcoded `3.3`. | D.3 | **TERBUKA** — di luar cakupan |
| 9 | Ambang kritis 54/250 hanya ada di `PatientState`; lapisan lain memakai 70/180. | E.3 | **SELESAI (Tugas 3)** — `src/constants.py` |
| 10 | Pengklasifikasi kondisi adalah RandomForestClassifier terlatih yang menimpa pengambangan regresi. | E.7 | **TERBUKA** — dokumentasi, bukan cacat; perlu diperjelas di Bab III |
| 11 | **Nomor halaman tidak disimpan sama sekali.** | F.4, G.6 | **SELESAI (Tugas 1 & 1B)** — `halaman_pdf` + `halaman_cetak`, tampil di UI |
| 12 | Re-ingest menghapus dan membangun ulang indeks tiap run; tanpa cache ekstraksi. | F.5 | **TERBUKA (disengaja)** — determinisme dipilih di atas kecepatan; biayanya < 1 menit |
| 13 | `_llm_context()` tidak pernah sampai ke LLM. | G.5 | **SELESAI (perbaikan lanjutan)** — diteruskan lewat `clinical_context` |
| 14 | `question_payload` menyebut "Prediksi 1 jam" padahal horizon +30 menit. | G.5 | **SELESAI (perbaikan lanjutan)** — horizon dari config; ternyata KUERI RETRIEVAL juga salah |
| 15 | `temperature` efektif 0.1 bukan 0.2; `max_tokens` tidak diterapkan. | G.4 | **SELESAI (Tugas 4)** — 0.2 dan 700, keduanya dari config |
| 16 | Fallback keyword juga menurunkan korpus ke ~30 chunk manual_kb. | G.7 | **TERBUKA (disengaja)** — perilaku fallback, kini terdokumentasi |
| 17 | Tidak ada reranking. | G.3 | **TERBUKA (disengaja)** — MMR dipertahankan agar angka lama sebanding |
| 18 | `WhatIfSimulator` diinstansiasi tetapi tidak pernah dipakai; seluruh method-nya kode mati. | H.5 | **SELESAI (Tugas 2)** — seluruh fitur dihapus |
| 19 | Cabang urgensi `medium` lewat `trend_rate == "moderate"` tidak dapat tercapai. | E.5 | **SELESAI (perbaikan lanjutan)** — `medium` = tren moderat ATAU stres tinggi |
| 20 | Data dari halaman Input Logbook tidak mengalir ke halaman prediksi. | H.4 | **TERBUKA** — di luar cakupan |
| 21 | `similarity` selalu `0.0`. | G.2 | **SELESAI (Tugas 1B)** — skor nyata, `Optional[float]` |
| 22 | Sebagian besar blok `rag:` di `config.yaml` tidak dibaca kode. | I.1 | **SELESAI (Tugas 4)** — tidak ada lagi parameter yang diabaikan |

**Temuan baru yang muncul selama refaktor:**

| # | Temuan | Bagian | Status |
|---|--------|--------|--------|
| 23 | **Jendela latih melintasi jeda sensor.** `create_sequences()` membentuk jendela per posisi baris; 5,3% jendela melompati jeda hingga 118 jam dan memperlakukannya sebagai satu langkah 5 menit. | A.5, B | **SELESAI (Tugas 5)** |
| 24 | **`_build_embeddings` terduplikasi** di `knowledge_base.py` dan `retriever.py`, keduanya dibaca saat import. Bila hanya satu diubah, embedding dokumen dan kueri memakai model berbeda dan retrieval merosot jadi derau **tanpa error apa pun**. | F.7 | **SELESAI (Tugas 4)** |
| 25 | **Koleksi Chroma memakai ruang jarak `l2`** (default), membuat skor relevansi bisa negatif. | F.5, G.2 | **SELESAI (Tugas 1)** — `hnsw:space=cosine` |
| 26 | **PDF yang gagal diekstrak dilewati diam-diam**, sehingga korpus bisa tidak lengkap tanpa peringatan. | F.2 | **SELESAI (Tugas 1)** — kini abort |
| 27 | **`eval_retrieval_crossfold.py` tidak membuat direktori keluarannya**, sehingga komputasi 6 fold (>1 jam) hilang di baris terakhir bila direktorinya belum ada. | — | **SELESAI (Tugas 7)** — `mkdir` di awal `main()`, gagal cepat |
| 28 | **Kuota Gemini di komentar config keliru ~50×** ("20 request/hari" vs 1.000 RPD sebenarnya), menyebabkan penghematan metrik RAGAS yang tidak perlu. | G.4 | **SELESAI (Tugas 6)** |
| 29 | **Seluruh kueri retrieval produksi menanyakan horizon 60 menit** padahal bundle memprediksi 30 menit dan UI menampilkan 30 menit. Skrip evaluasi pun terbelah: ablation memakai 60, crossfold/realcases memakai 30, atas korpus dan kasus yang sama. | G.1, G.5 | **SELESAI (perbaikan lanjutan)** |
| 30 | **`build_question_payload` selalu mengirim `Risk: N/A`** ke LLM pada jalur produksi — cabang yang mengisi `risk_level` hanya aktif bila `prediction` berupa dict, sedangkan pipeline selalu mengirim float. LLM tidak pernah menerima status risiko. | G.5 | **SELESAI (perbaikan lanjutan)** |
| 31 | **KB sementara pada sweep `chunk_size` tidak menerapkan filter fragmen < 80 karakter** yang dipakai `reingest_kb.py`, menghasilkan 2.202 chunk vs 2.061 di produksi. Perbandingan antar-`chunk_size` tetap sah, tetapi angka absolutnya tidak sebanding dengan produksi. | F.6 | **TERBUKA** |
| 32 | **Perancu evaluasi Tugas 5:** himpunan uji ikut menyusut saat segmentasi jeda diterapkan, sehingga perbaikan RMSE tidak dapat dipisahkan antara "model lebih baik" dan "kasus uji tak sah hilang". | C.1 | **TERBUKA** — perlu run terkontrol ~16 mnt |

**Status akhir: 16 dari 22 temuan Versi 1 selesai; 10 temuan baru ditemukan selama
refaktor, 8 di antaranya selesai.**

Temuan baru yang paling berbahaya karena **tidak memberi error sama sekali**:
- **#24** — model embedding dokumen dan kueri dapat menyimpang tanpa peringatan
- **#27** — komputasi 8 jam hilang di baris terakhir karena direktori keluaran belum ada
- **#29** — setiap kueri produksi menanyakan horizon yang salah selama ini

**Catatan penting untuk temuan #4.** Batas interpolasi memang ditambahkan, tetapi
pada `ohio_t1dm_merged.csv` efeknya **nol**: berkas itu tidak memuat satu pun NaN.
Baris yang hilang akibat jeda sensor memang **tidak ada** di dalam berkas, bukan hadir
sebagai NaN — sehingga `interpolate()` tidak pernah menyentuhnya. Premis temuan #4
karena itu tidak sepenuhnya tepat; cacat yang sesungguhnya adalah temuan #23, dan
itulah yang ditangani segmentasi jendela. Batas interpolasi tetap dipasang sebagai
pengaman untuk `manual_logbook.csv` yang memang dapat memuat NaN.
