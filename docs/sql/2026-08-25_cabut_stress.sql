-- =====================================================================
-- Pencabutan total variabel stres — Supabase
-- Tanggal: 25 Agustus 2026
-- =====================================================================
--
-- LATAR
-- -----
-- `stress` tidak pernah menjadi fitur model: ia tidak ada pada
-- config.model.engineered_features, dan kanal `stressors` OhioT1DM hanya
-- memuat 7 event di seluruh 12 pasien sehingga inheren tak informatif.
-- Penulisan ke `stress_events` sudah dihentikan lebih dulu; jalur bacanya
-- kemudian dicabut dari backend/services/supabase_data_service.py.
--
-- URUTAN INI MENGIKAT
-- -------------------
--   1. jalur baca dicabut dari kode        <- SUDAH
--   2. backend di-restart                  <- LAKUKAN SEBELUM SKRIP INI
--   3. skrip ini dijalankan
--
-- Menjalankan skrip ini sebelum langkah 2 akan membuat setiap permintaan
-- penilaian klinis gagal dengan HTTP 500, karena jalur baca yang lama
-- masih menanyakan tabel yang sudah tidak ada.
--
-- CATATAN
-- -------
-- `drop table` menghapus data secara permanen dan tidak dapat dibatalkan.
-- Jalankan blok PEMERIKSAAN lebih dulu dan baca hasilnya sebelum
-- menjalankan blok PENCABUTAN.
-- =====================================================================


-- ---------------------------------------------------------------------
-- 1. PEMERIKSAAN — jalankan sendiri, baca hasilnya, jangan dilewati
-- ---------------------------------------------------------------------

-- 1a. Berapa baris yang akan hilang?
select count(*) as jumlah_baris
from public.stress_events;

-- 1b. Rentang waktunya, untuk memastikan tidak ada penulisan baru
--     setelah penulisan seharusnya berhenti.
select
    min(timestamp) as terlama,
    max(timestamp) as terbaru,
    count(distinct patient_id) as jumlah_pasien
from public.stress_events;

-- 1c. Adakah objek lain yang bergantung padanya (foreign key, view)?
--     Bila ada hasil, tangani lebih dulu — jangan pakai `cascade` membabi
--     buta, karena ia ikut menghapus objek yang mungkin masih dipakai.
select
    tc.constraint_name,
    tc.table_name as tabel_yang_merujuk
from information_schema.table_constraints tc
join information_schema.constraint_column_usage ccu
    on tc.constraint_name = ccu.constraint_name
where tc.constraint_type = 'FOREIGN KEY'
  and ccu.table_name = 'stress_events';

-- 1d. Adakah kolom bernama `stress*` yang menempel pada tabel lain?
--     Tabel yang benar-benar dirujuk kode: patients, glucose_sources,
--     glucose_readings, insulin_events, meal_events, activity_events,
--     clinical_assessments.
select
    table_name,
    column_name,
    data_type
from information_schema.columns
where table_schema = 'public'
  and column_name ilike '%stress%'
order by table_name, column_name;


-- ---------------------------------------------------------------------
-- 2. PENCABUTAN — jalankan hanya setelah blok 1 dibaca
-- ---------------------------------------------------------------------

drop table if exists public.stress_events;

-- Bila blok 1d menemukan kolom `stress*` pada tabel lain, cabut satu per
-- satu di sini. Contoh (SESUAIKAN nama tabel dan kolomnya dengan hasil
-- nyata; jangan jalankan sebagai tebakan):
--
--     alter table public.glucose_readings drop column if exists stress_level;


-- ---------------------------------------------------------------------
-- 3. VERIFIKASI SESUDAHNYA
-- ---------------------------------------------------------------------

-- Harus mengembalikan nol baris.
select
    table_name,
    column_name
from information_schema.columns
where table_schema = 'public'
  and (table_name ilike '%stress%' or column_name ilike '%stress%');
