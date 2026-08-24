"use client";

import { useEffect, useRef, useState } from "react";
import "./dashboard.css";

import {
  runClinicalAssessment,
  type Observation,
  type ClinicalResponse,
  type GlucoseSource,
  type VerifikasiSitasi,
} from "../lib/api";

import { getAssessmentHistory } from "../lib/assessment";

import GlucoseChart from "../components/GlucoseChart";

import {
  getGlucoseObservations,
  getPatients,
  getPatientSources,
  getLatestPatientContext,
  type GlucosePoint,
} from "../lib/glucose";

import Navbar from "../components/Navbar";
import Footer from "../components/Footer";

type Citation = {
  source?: string;
  title?: string;
  year?: number | string;
  page?: number | string;
  kb_id?: string;
  // Medan di bawah dikirim backend sejak 24 Agustus 2026 supaya dokter dapat
  // membaca kalimat yang benar-benar dikutip, bukan hanya identitas dokumen.
  //
  // PENTING: `snippet` SUDAH dipotong di sisi Python oleh
  // src/rag/citations.py (batas kalimat, empat penjaga, teruji). JANGAN memotong
  // ulang di sini — aturan yang disalin akan menyimpang dan tesnya tidak akan
  // menangkapnya.
  rank?: number;
  nama_dokumen?: string;
  snippet?: string;
  teks_lengkap?: string;
  n_char?: number;
  snippet_terpotong?: boolean;
  cara_potong?: string;
  mulai_kalimat_utuh?: boolean;
  akhir_kalimat_utuh?: boolean;
  chunk_id?: string;
};

type AssessmentHistoryItem = {
  id: number;
  patient_id: string;
  assessed_at: string;
  current_glucose: number | null;
  prediction_30m: number | null;
  prediction_60m: number | null;
  condition: string | null;
  glucose_source: GlucoseSource | null;
  recommendation: string | null;
  grounded: boolean;
  sources: Citation[];
};

type Advisory = {
  risk_level: string;
  explanation: string;
  grounded: boolean;
  // False bila narasi berasal dari templat karena model bahasa tidak tersedia.
  narasiLlm: boolean;
  citations: Citation[];
  verifikasiSitasi?: VerifikasiSitasi | null;
  timings?: Record<string, number> | null;
};


// =============================================================
// RENDER SBAR
// =============================================================
//
// Model bahasa diminta menjawab dalam lima bagian berjudul (src/rag/prompts.py).
// Sebelum ini seluruh jawaban dijejalkan ke SATU elemen <p>, sehingga kelima judul
// menyatu menjadi satu gumpalan teks DAN tanda bintang penebalan ikut terbaca
// mentah oleh dokter. Struktur yang susah payah diminta di prompt hilang tepat di
// langkah terakhir sebelum sampai ke mata pembaca.
//
// Judul di bawah HARUS sama persis dengan yang tertulis pada SYSTEM_PROMPT. Bila
// prompt diubah, ubah daftar ini juga - keduanya adalah satu kontrak.
const JUDUL_SBAR = [
  "Situasi",
  "Latar",
  "Penilaian",
  "Rekomendasi",
  "Yang tidak dapat disimpulkan dari data ini",
] as const;

type BagianSBAR = { judul: string; isi: string };

/** Pecah jawaban menjadi bagian-bagian SBAR. Kembalikan null bila polanya tidak ada. */
function pecahSBAR(teks: string): BagianSBAR[] | null {
  if (!teks) return null;

  // Judul boleh datang sebagai "**Situasi**" atau "Situasi" di awal baris. Kedua
  // bentuk ditoleransi: kepatuhan model tidak dijamin, dan kegagalan mengenali
  // judul TIDAK boleh berujung teks yang hilang — lihat fallback di komponen.
  const alternatif = JUDUL_SBAR.map((j) =>
    j.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
  ).join("|");

  const pola = new RegExp(
    "^\\s*(?:\\*\\*)?\\s*(" + alternatif + ")\\s*(?:\\*\\*)?\\s*:?\\s*$",
    "i",
  );

  const bagian: BagianSBAR[] = [];
  let aktif: BagianSBAR | null = null;

  for (const baris of teks.split(/\r?\n/)) {
    const cocok = baris.match(pola);

    if (cocok) {
      aktif = { judul: cocok[1], isi: "" };
      bagian.push(aktif);
      continue;
    }

    if (aktif) aktif.isi += (aktif.isi ? "\n" : "") + baris;
  }

  // Butuh minimal dua bagian berisi untuk yakin ini memang keluaran SBAR.
  const berisi = bagian.filter((x) => x.isi.trim().length > 0);
  return berisi.length >= 2 ? berisi : null;
}

/** Ubah penanda [S1], [S2] menjadi chip yang menggulir ke rujukannya. */
function denganChipSitasi(
  teks: string,
  keSumber: (n: number) => void,
): React.ReactNode[] {
  const potongan: React.ReactNode[] = [];
  const pola = /\[S(\d+)\]/g;

  let akhir = 0;
  let cocok: RegExpExecArray | null;

  while ((cocok = pola.exec(teks)) !== null) {
    if (cocok.index > akhir) potongan.push(teks.slice(akhir, cocok.index));

    const nomor = Number(cocok[1]);

    potongan.push(
      <button
        key={`${cocok.index}-S${nomor}`}
        type="button"
        className="sitasi-chip"
        onClick={() => keSumber(nomor)}
        title={`Buka sumber ${nomor}`}
      >
        S{nomor}
      </button>,
    );

    akhir = cocok.index + cocok[0].length;
  }

  if (akhir < teks.length) potongan.push(teks.slice(akhir));

  return potongan;
}

function IsiBagian({
  isi,
  keSumber,
}: {
  isi: string;
  keSumber: (n: number) => void;
}) {
  // Daftar berjenjang "a. b. c." pada Rekomendasi dirender sebagai daftar urut.
  // Sisanya paragraf biasa. Penebalan **...** DILUCUTI, bukan dirender: tanda
  // bintang mentah itulah cacat yang sedang diperbaiki, dan memasang parser
  // markdown penuh menjelang evaluasi dokter bukan pertukaran yang sepadan.
  const baris = isi
    .split(/\r?\n/)
    .map((b) => b.replace(/\*\*/g, "").trim())
    .filter(Boolean);

  const adalahButir = (b: string) => /^[a-z][.)]\s/i.test(b);

  const butir = baris.filter(adalahButir);

  if (butir.length >= 2) {
    const lain = baris.filter((b) => !adalahButir(b));

    return (
      <>
        {lain.map((b, i) => (
          <p key={`p${i}`} className="advisory-text">
            {denganChipSitasi(b, keSumber)}
          </p>
        ))}

        <ol className="sbar-daftar">
          {butir.map((b, i) => (
            <li key={`l${i}`}>
              {denganChipSitasi(b.replace(/^[a-z][.)]\s*/i, ""), keSumber)}
            </li>
          ))}
        </ol>
      </>
    );
  }

  return (
    <>
      {baris.map((b, i) => (
        <p key={`p${i}`} className="advisory-text">
          {denganChipSitasi(b, keSumber)}
        </p>
      ))}
    </>
  );
}

function AdvisorySBAR({
  teks,
  keSumber,
}: {
  teks: string;
  keSumber: (n: number) => void;
}) {
  const bagian = pecahSBAR(teks);

  // CADANGAN. Jalur templat tidak berbentuk SBAR, dan model bahasa pun bisa
  // tidak patuh. Dalam kedua keadaan itu teks ditampilkan APA ADANYA dengan
  // baris baru dipertahankan. Tidak pernah ada teks yang dibuang hanya karena
  // bentuknya tidak dikenali — kehilangan isi jauh lebih buruk daripada
  // kehilangan struktur.
  if (!bagian) {
    return (
      <p className="advisory-text advisory-text--praformat">
        {denganChipSitasi(teks, keSumber)}
      </p>
    );
  }

  return (
    <div className="sbar">
      {bagian.map((x) => (
        <section key={x.judul} className="sbar-bagian">
          <h4 className="sbar-judul">{x.judul}</h4>
          <IsiBagian isi={x.isi} keSumber={keSumber} />
        </section>
      ))}
    </div>
  );
}


export default function Home() {
  // =========================================================
  // STATE
  // =========================================================

  const [loading, setLoading] = useState(false);

  const [error, setError] =
    useState<string | null>(null);

  const [historyLoading, setHistoryLoading] = useState(true);

  const [result, setResult] =
    useState<ClinicalResponse | null>(null);

  const [history, setHistory] =
    useState<AssessmentHistoryItem[]>([]);

  const [glucoseHistory, setGlucoseHistory] =
    useState<GlucosePoint[]>([]);

  // Stage 4: explicit glucose input modality.
  const [glucoseSource, setGlucoseSource] =
    useState<GlucoseSource>("CGM");

  const [uiStateReady, setUiStateReady] =
    useState(false);

  // =========================================================
  // DEMO / CURRENT PATIENT DATA
  // =========================================================

  const [patientId, setPatientId] = useState("");
  const [patientOptions, setPatientOptions] =
    useState<string[]>([]);
  const [sourceOptions, setSourceOptions] =
    useState<GlucoseSource[]>([]);
  const [observations, setObservations] =
    useState<Observation[]>([]);
  const [recentContext, setRecentContext] =
    useState({ insulin: 0, carbs: 0, activity: 0 });

  // Sakelar "tampilkan potongan dokumen secara utuh". Bawaannya RINGKAS supaya
  // panel rujukan tidak mendominasi layar, tetapi dokter dapat membuka teks penuh
  // ketika ingin memverifikasi sendiri apakah rekomendasi berpijak pada dokumen.
  const [kutipanUtuh, setKutipanUtuh] = useState(false);
  const [rujukanTerbuka, setRujukanTerbuka] = useState(true);

  // Pertanyaan klinis yang diketik dokter. Kosong = pakai kalimat bawaan.
  // Endpoint sudah menerima `question` sejak awal dan _build_query sudah
  // memprioritaskannya; yang belum ada hanyalah tempat mengetiknya. Pertanyaan
  // generik adalah salah satu sebab jawaban terasa generik.
  const [pertanyaan, setPertanyaan] = useState("");

  // Wadah tiap rujukan, supaya chip [S1] dapat menggulir ke sumbernya.
  const sumberRefs = useRef<Array<HTMLDivElement | null>>([]);

  const keSumber = (nomor: number) => {
    setRujukanTerbuka(true);

    // Panel rujukan mungkin baru saja dibuka pada baris di atas, jadi penggulirannya
    // ditunda sampai React selesai memasang elemennya.
    //
    // setTimeout, BUKAN requestAnimationFrame. rAF tidak pernah dipanggil ketika
    // halaman tidak sedang digambar (tab latar, jendela tertutup, pratinjau yang
    // tidak ditampilkan), sehingga penggulirannya hilang diam-diam persis pada
    // keadaan yang paling sulit disadari. setTimeout tetap berjalan; ia hanya
    // diperlambat. Pembaruan state React sendiri dituntaskan pada microtask,
    // yang selalu selesai sebelum timeout ini berjalan.
    window.setTimeout(() => {
      const wadah = sumberRefs.current[nomor - 1];

      if (!wadah) return;

      // Penggulirannya mulus HANYA bila animasi memang dikehendaki dan mungkin.
      // Animasi mulus tidak berjalan pada halaman yang tidak sedang digambar, dan
      // tidak dikehendaki oleh pembaca yang menyetel prefers-reduced-motion; pada
      // kedua keadaan itu lompatan langsung jauh lebih baik daripada tidak
      // bergerak sama sekali.
      const mulus =
        !document.hidden &&
        !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

      wadah.scrollIntoView({
        behavior: mulus ? "smooth" : "auto",
        block: "center",
      });

      wadah.classList.add("rujukan-item--disorot");

      window.setTimeout(
        () => wadah.classList.remove("rujukan-item--disorot"),
        1600,
      );
    }, 0);
  };

  // =========================================================
  // CURRENT GLUCOSE / CONTEXT
  // =========================================================

  const currentGlucose =
    observations[observations.length - 1]?.glucose ?? null;

  const currentObservation =
    observations[observations.length - 1];

  const insulin = recentContext.insulin;

  const carbs = recentContext.carbs;

  const activity = recentContext.activity;

  // =========================================================
  // FORECAST
  // =========================================================

  // Never fabricate a forecast when the backend has not produced one.
  const prediction30 =
    result?.prediction?.prediction_30m ??
    null;

  const prediction60 =
    result?.prediction?.horizons?.find(
      (item) => item.horizon_minutes === 60,
    )?.prediction ?? null;

  const primaryHorizon =
    result?.prediction?.horizons?.[0] ?? null;
  const isFingerStick = glucoseSource === "FINGER_STICK";
  const primaryPrediction = isFingerStick
    ? primaryHorizon?.prediction ?? null
    : prediction30;
  const primaryHorizonLabel = isFingerStick
    ? `${primaryHorizon?.horizon_minutes ?? 240} MIN`
    : "30 MIN";

  const predictionAvailable =
    result?.prediction?.prediction_available ??
    (prediction30 !== null || prediction60 !== null);

  const sourceDefaultMinimum =
    glucoseSource === "FINGER_STICK" ? 8 : 12;

  // =========================================================
  // LOAD ASSESSMENT HISTORY
  // =========================================================

  const loadHistory = async () => {
    try {
      const data =
        (await getAssessmentHistory(patientId)) as AssessmentHistoryItem[];

      const safeData = data ?? [];
      setHistory(safeData);

    } catch (err) {
      console.error(
        "Failed to load assessment history:",
        err,
      );
    } finally {
      setHistoryLoading(false);
    }
  };

  // =========================================================
  // LOAD GLUCOSE HISTORY
  // =========================================================

  const loadPatientsAndObservations = async () => {
    try {
      const patientRecords = await getPatients();
      const patients = patientRecords.map((item) => item.patient_code);
      setPatientOptions(patients);

      const storedPatient = window.localStorage.getItem(
        "last_patient_id",
      );
      const nextPatient =
        patientId && patients.includes(patientId)
          ? patientId
          : storedPatient && patients.includes(storedPatient)
            ? storedPatient
            : patients[0] ?? "";

      if (nextPatient !== patientId) {
        setPatientId(nextPatient);
      }

      const sources = nextPatient
        ? await getPatientSources(nextPatient)
        : [];
      const availableSources = sources.map((item) => item.source_type);
      setSourceOptions(availableSources);

      const nextSource = availableSources.includes(glucoseSource)
        ? glucoseSource
        : availableSources[0];
      if (nextSource && nextSource !== glucoseSource) {
        setGlucoseSource(nextSource);
        return;
      }

      if (!nextPatient) {
        setObservations([]);
        setGlucoseHistory([]);
        return;
      }

      const nextObservations =
        await getGlucoseObservations(
          nextPatient,
          glucoseSource,
        );
      setObservations(nextObservations);
      setRecentContext(
        await getLatestPatientContext(nextPatient),
      );
      setGlucoseHistory(
        nextObservations.map((item) => ({
          timestamp: item.timestamp,
          glucose: item.glucose,
        })),
      );
    } catch (err) {
      console.error(
        "Failed to load patients and observations:",
        err,
      );
      setError(
        "Data pasien gagal dimuat. Periksa koneksi Supabase dan RLS policy.",
      );
      setPatientOptions([]);
      setObservations([]);
      setGlucoseHistory([]);
    }
  };

  // =========================================================
  // LOAD DATA ON PAGE LOAD
  // =========================================================

  useEffect(() => {
    const storedPatient = window.localStorage.getItem(
      "last_patient_id",
    );
    const storedSource = window.localStorage.getItem(
      "last_glucose_source",
    );

    if (storedPatient) {
      setPatientId(storedPatient);
    }

    if (
      storedSource === "CGM" ||
      storedSource === "FINGER_STICK"
    ) {
      setGlucoseSource(storedSource);
    }

    setUiStateReady(true);
  }, []);

  useEffect(() => {
    if (!uiStateReady) {
      return;
    }

    if (patientId) {
      window.localStorage.setItem(
        "last_patient_id",
        patientId,
      );
    }
    window.localStorage.setItem(
      "last_glucose_source",
      glucoseSource,
    );
  }, [uiStateReady, patientId, glucoseSource]);

  useEffect(() => {
    if (uiStateReady) {
      loadHistory();
      loadPatientsAndObservations();
    }
  }, [uiStateReady, patientId, glucoseSource]);

  const minimumRequired =
    result?.prediction?.minimum_required ?? sourceDefaultMinimum;
  const validReadingCount = observations.filter(
    (observation) => Number.isFinite(observation.glucose),
  ).length;
  const hasSufficientHistory =
    validReadingCount >= minimumRequired;

  const dataAvailability =
    patientOptions.length === 0 || !patientId || observations.length === 0
      ? "NO_DATA"
      : hasSufficientHistory
        ? "READY"
        : "INSUFFICIENT_HISTORY";

  const clearAssessmentContext = () => {
    setResult(null);
    setError(null);
    setObservations([]);
    setGlucoseHistory([]);
    setRecentContext({ insulin: 0, carbs: 0, activity: 0 });
  };

  // =========================================================
  // RUN CLINICAL ASSESSMENT
  // =========================================================

  const handleRunClinical = async () => {
    if (!patientId || observations.length === 0) {
      setError(
        `Belum ada data ${glucoseSource} untuk pasien ini. Input data melalui Logbook terlebih dahulu.`,
      );
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data =
        await runClinicalAssessment(
          patientId,
          pertanyaan.trim() ||
            (predictionAvailable
              ? "Berikan rekomendasi klinis berdasarkan hasil prediksi glukosa pasien."
              : "Bagaimana kondisi pasien saat ini berdasarkan glukosa terakhir?"),
            observations,
          glucoseSource,
        );

      setResult(data);

      await loadHistory();
      await loadPatientsAndObservations();
    } catch (err) {
      console.error(
        "Clinical assessment failed:",
        err,
      );

      setError(
        err instanceof Error
          ? err.message
          : "Clinical assessment failed.",
      );
    } finally {
      setLoading(false);
    }
  };

  // =========================================================
  // CREATE ADVISORY FROM LATEST SUPABASE ASSESSMENT
  // =========================================================

  const advisory: Advisory | null = result?.clinical_advisory
    ? {
          risk_level:
            result.clinical_advisory.risk_level ??
            "AMAN",

          explanation:
            result.clinical_advisory.explanation ??
            "No clinical recommendation available.",

          grounded:
            result.clinical_advisory.grounded,

          // Bawaan true supaya respons lama (tanpa medan ini) tidak salah
          // dilaporkan sebagai templat.
          narasiLlm:
            result.clinical_advisory.narasi_llm ?? true,

          citations:
            result.clinical_advisory.citations ?? [],

          verifikasiSitasi:
            result.clinical_advisory.verifikasi_sitasi ?? null,

          timings:
            result.clinical_advisory.timings ?? null,
        }
    : null;

  // =========================================================
  // RENDER
  // =========================================================

    return (
    <>

      {/* ===================================================
          GLOBAL NAVBAR
          =================================================== */}

      <Navbar />


      {/* ===================================================
          MAIN DASHBOARD
          =================================================== */}

      <main className="dashboard">


        {/* =================================================
            HEADER
            ================================================= */}

        <header className="topbar">

          <div>

            <p className="eyebrow">
              Clinical Decision Support
            </p>

            <h1>
              Glucose Monitor
            </h1>

          </div>


          <div className="patient-badge">

            <span className="status-dot" />

            Patient {patientId}

          </div>

        </header>


        {/* =================================================
            INPUT MODALITY
            ================================================= */}

        <section className="section">
          <div className="section-heading">
            <div>
              <p className="eyebrow">
                Input modality
              </p>

              <h2>
                Glucose source
              </h2>
            </div>
          </div>

          <div
            style={{
              display: "flex",
              gap: 16,
              alignItems: "center",
              flexWrap: "wrap",
              marginBottom: 16,
            }}
          >
            <label htmlFor="patient-selector">
              Patient
            </label>
            <select
              id="patient-selector"
              value={patientId}
              onChange={(event) => {
                setPatientId(event.target.value);
                setSourceOptions([]);
                clearAssessmentContext();
              }}
            >
              {patientOptions.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
            <span className="muted">
              Data availability · {dataAvailability}
            </span>
          </div>

          <div
            style={{
              display: "flex",
              gap: 16,
              alignItems: "center",
              flexWrap: "wrap",
            }}
          >
            {sourceOptions.map((source) => (
            <label
              key={source}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                cursor: "pointer",
              }}
            >
              <input
                type="radio"
                name="glucose-source"
                value={source}
                checked={glucoseSource === source}
                onChange={() => {
                  setGlucoseSource(source);
                  clearAssessmentContext();
                }}
              />

              <span>{source === "CGM" ? "CGM" : "Finger-stick / SMBG"}</span>
            </label>
            ))}
            {sourceOptions.length === 0 && (
              <span className="muted">No glucose source available.</span>
            )}

            <span className="muted">
              Active source · {glucoseSource}
            </span>
          </div>
        </section>


        {/* =================================================
            CURRENT GLUCOSE
            ================================================= */}

        <section className="hero-card">

          <div>

            <p className="label">
              Current glucose
            </p>


            <div className="glucose-value">

              {currentGlucose !== null
                ? currentGlucose.toFixed(1)
                : "—"}

              <span>
                mg/dL
              </span>

            </div>


            <p className="muted">
              Last measurement · {currentObservation?.timestamp ?? "—"} · {glucoseSource}
            </p>

          </div>


          <div className="risk normal">

            <span className="risk-dot" />

            {result?.clinical_advisory?.risk_level ??
              advisory?.risk_level ??
              "NO ASSESSMENT"}

          </div>

        </section>


        {/* =================================================
            FORECAST
            ================================================= */}

        <section className="section">

          <div className="section-heading">

            <div>

              <p className="eyebrow">
                Forecast
              </p>

              <h2>
                Glucose prediction
              </h2>

            </div>


            <div className="pertanyaan-blok">

              {/*
                Pertanyaan klinis bebas. Sebelumnya SATU kalimat dipatri untuk
                setiap pemanggilan, sehingga tiap pasien menghasilkan kueri
                penelusuran yang nyaris sama dan jawaban yang terasa seragam.
                Pertanyaan yang spesifik adalah pengungkit terbesar melawan
                jawaban generik, karena ia mengubah kueri retrieval sekaligus
                fokus penalaran.
              */}
              <label className="pertanyaan-label" htmlFor="pertanyaan">
                Pertanyaan klinis (opsional)
              </label>

              <input
                id="pertanyaan"
                className="pertanyaan-input"
                type="text"
                value={pertanyaan}
                onChange={(e) => setPertanyaan(e.target.value)}
                placeholder="mis. Apakah dosis bolus perlu disesuaikan sebelum tidur?"
                disabled={loading}
              />

            <button
              className="primary-button"
              onClick={handleRunClinical}
              disabled={loading}
            >

              {loading
                ? "Assessing..."
                : "Run Clinical Assessment"}

            </button>

            </div>

          </div>


          {result && !predictionAvailable ? (

            <div className="advisory-card">

              <p className="eyebrow">
                Prediction unavailable
              </p>

              <h2>
                {dataAvailability === "NO_DATA"
                  ? "Source unavailable"
                  : "Current-state assessment only"}
              </h2>

              <p className="advisory-text">
                {dataAvailability === "NO_DATA"
                  ? `Belum ada data ${glucoseSource} untuk pasien ini. Input data ${glucoseSource} melalui Logbook untuk melihat data pasien menggunakan sumber ini.`
                  : result?.prediction.reason ??
                  result?.prediction.history?.reason ??
                  `Insufficient historical ${glucoseSource} observations: ${validReadingCount}/${minimumRequired}.`}
              </p>

              <p className="advisory-disclaimer">
                Current glucose is not treated as a future
                prediction. Add sufficient historical data
                to enable the forecasting model.
              </p>

            </div>

          ) : dataAvailability === "READY" && !result ? (
            <div className="advisory-card">
              <p className="eyebrow">Prediction ready</p>
              <p className="advisory-text">
                Historical data is sufficient. Run Clinical Assessment to generate prediction.
              </p>
            </div>
          ) : dataAvailability !== "READY" ? (
            <div className="advisory-card">
              <p className="eyebrow">Prediction unavailable</p>
              <p className="advisory-text">
                {dataAvailability === "NO_DATA"
                  ? `Belum ada data ${glucoseSource} untuk pasien ini.`
                  : `Insufficient historical ${glucoseSource} observations: ${validReadingCount}/${minimumRequired}.`}
              </p>
            </div>
          ) : (

            <>
              {/* Forecast timeline */}

              <div className="forecast-card">

                <div className="forecast-track">


                  {/* NOW */}

                  <div className="forecast-point current">

                    <div className="forecast-dot" />

                    <div className="forecast-info">

                      <span className="forecast-time">
                        NOW
                      </span>

                      <strong>
                        {currentGlucose !== null
                          ? currentGlucose.toFixed(1)
                          : "—"}
                      </strong>

                      <span>
                        mg/dL
                      </span>

                    </div>

                  </div>


                  <div className="forecast-line" />


                  {/* 30 MIN */}

                  <div className="forecast-point">

                    <div className="forecast-dot" />

                    <div className="forecast-info">

                      <span className="forecast-time">
                        {primaryHorizonLabel}
                      </span>

                      <strong>
                        {primaryPrediction !== null
                          ? primaryPrediction.toFixed(1)
                          : "—"}
                      </strong>

                      <span>
                        mg/dL
                      </span>

                    </div>

                  </div>


                  {!isFingerStick && (
                    <>
                      <div className="forecast-line" />

                      {/* 60 MIN */}

                      <div className="forecast-point">

                    <div className="forecast-dot" />

                    <div className="forecast-info">

                      <span className="forecast-time">
                        60 MIN
                      </span>

                      <strong>
                        {!isFingerStick && prediction60 !== null
                          ? prediction60.toFixed(1)
                          : "—"}
                      </strong>

                      <span>
                        mg/dL
                      </span>

                    </div>

                      </div>
                    </>
                  )}

                </div>


                {/* Forecast summary */}

                <div className="forecast-summary">

                  <div>

                    <span className="label">
                      Current
                    </span>

                    <strong>
                      {currentGlucose !== null
                        ? currentGlucose.toFixed(1)
                        : "—"}
                      {" "}
                      mg/dL
                    </strong>

                  </div>


                  <div>

                    <span className="label">
                      {isFingerStick
                        ? `${primaryHorizon?.horizon_minutes ?? 240}-minute forecast`
                        : "30-minute forecast"}
                    </span>

                    <strong>
                      {primaryPrediction !== null
                        ? `${primaryPrediction.toFixed(1)} mg/dL`
                        : "Not available"}
                    </strong>

                  </div>


                  {!isFingerStick && (
                    <div>
                      <span className="label">
                        60-minute forecast
                      </span>

                      <strong>
                        {prediction60 !== null
                          ? `${prediction60.toFixed(1)} mg/dL`
                          : "Not available"}
                      </strong>
                    </div>
                  )}

                </div>

              </div>
            </>

          )}

        </section>


        {/* =================================================
            PATIENT CONTEXT
            ================================================= */}

        <section className="section">

          <div className="section-heading">

            <div>

              <p className="eyebrow">
                Patient context
              </p>

              <h2>
                Recent state
              </h2>

            </div>

          </div>


          <div className="context-grid">

            <ContextCard
              title="Insulin"
              value={Number(
                insulin,
              ).toFixed(1)}
              unit="units"
            />

            <ContextCard
              title="Carbohydrate"
              value={String(
                carbs,
              )}
              unit="g"
            />

            <ContextCard
              title="Activity"
              value={String(
                activity,
              )}
              unit="level"
            />

          </div>

        </section>


        {/* =================================================
            GLUCOSE HISTORY
            ================================================= */}

        <section className="section">

          <div className="glucose-history-card">

            <div className="section-heading">

              <div>

                <p className="eyebrow">
                  Glucose history
                </p>

                <h2>
                  Recent glucose trend
                </h2>

              </div>


              <span className="muted">
                Last {glucoseHistory.length}
                {" "}
                measurements
              </span>

            </div>


            {glucoseHistory.length > 0 ? (

              <GlucoseChart
                data={glucoseHistory}
              />

            ) : (

              <div className="empty-chart">

                No glucose data available.

              </div>

            )}

          </div>

        </section>


        {/* =================================================
            CLINICAL ADVISORY
            ================================================= */}

        <section className="advisory-card">


          <div className="advisory-header">


            <div className="advisory-icon">

              {advisory
                ? "✓"
                : "—"}

            </div>


            <div className="advisory-title">

              <p className="eyebrow">
                Clinical advisory
              </p>

              <h2>

                {advisory
                  ? result?.mode === "current_state"
                    ? "Current-state assessment"
                    : "Assessment available"
                  : "No assessment yet"}

              </h2>

            </div>


            {advisory && (

              <span
                className={`risk-badge ${
                  advisory.risk_level
                    ?.toLowerCase() ===
                  "bahaya"
                    ? "danger"
                    : advisory.risk_level
                          ?.toLowerCase() ===
                      "waspada"
                      ? "warning"
                      : "safe"
                }`}
              >

                {advisory.risk_level
                  ?.toUpperCase() ||
                  "AMAN"}

              </span>

            )}

          </div>


          {advisory ? (

            <div className="advisory-content">


              {/* Clinical assessment */}

              <div className="advisory-section">

                <p className="advisory-label">
                  Clinical assessment
                </p>

                <AdvisorySBAR
                  teks={advisory.explanation}
                  keSumber={keSumber}
                />

              </div>


              {/* Evidence */}

              <div className="advisory-section">

                <p className="advisory-label">
                  Evidence status
                </p>

                {/*
                  Peringatan asal-usul narasi. Rujukan di bawah tetap sah — penelusuran
                  tidak ikut gagal ketika model bahasa gagal — tetapi kalimat penjelas
                  di atasnya berasal dari templat, bukan penalaran atas dokumen.
                  Membiarkan dokter mengira sebaliknya lebih buruk daripada tidak
                  menampilkan apa pun.
                */}
                {!advisory.narasiLlm && (
                  <p className="advisory-peringatan-templat">
                    Model bahasa sedang tidak tersedia. Teks penilaian di atas disusun
                    dari templat, <strong>bukan</strong> hasil penalaran atas dokumen.
                    Rujukan di bawah tetap berasal dari penelusuran pedoman dan dapat
                    dibaca sendiri.
                  </p>
                )}

                <p className="advisory-text">

                  {advisory.grounded
                    ? "Recommendation is grounded in the retrieved clinical knowledge base."
                    : "Retrieved evidence is insufficient for a grounded recommendation."}

                </p>

                {/*
                  Angka ber-penanda yang TIDAK dapat ditemukan pada potongan yang
                  ditunjuknya. Aturan prompt saja tidak cukup: bila penelusuran
                  meleset, model mengisi lubang dari ingatannya lalu tetap memberi
                  penanda, dan dokter yang membuka halaman itu tidak menemukan
                  apa-apa di sana.

                  DITAMPILKAN APA ADANYA: angka, sumber yang diklaim, kalimatnya.
                  TANPA persentase dan TANPA lencana kepercayaan - sebuah angka
                  "92% terverifikasi" mengundang pembaca memperlakukannya sebagai
                  ukuran mutu klinis, padahal ini hanya pencocokan tekstual.
                */}
                {advisory.verifikasiSitasi &&
                  advisory.verifikasiSitasi.n_tidak_terverifikasi > 0 && (
                    <div className="verifikasi-panel">
                      <p className="verifikasi-judul">
                        Angka berikut diberi penanda sumber, tetapi tidak
                        ditemukan pada potongan yang ditunjuk. Periksa sendiri
                        sebelum memakainya.
                      </p>

                      <ul className="verifikasi-daftar">
                        {advisory.verifikasiSitasi.temuan
                          .filter(
                            (t) => t.status === "TIDAK_TERVERIFIKASI",
                          )
                          .map((t, i) => (
                            <li key={`v${i}`}>
                              <code>{t.nilai}</code>
                              {t.penanda_diklaim
                                ? ` diklaim dari ${t.penanda_diklaim}`
                                : " tanpa sumber yang jelas"}
                              <span className="verifikasi-kalimat">
                                {t.kalimat}
                              </span>
                            </li>
                          ))}
                      </ul>
                    </div>
                  )}

                {/*
                  Durasi per tahap. Ditampilkan supaya penantian yang panjang
                  terbaca sebagai kerja yang benar-benar terjadi, dan supaya
                  terlihat tahap MANA yang lambat.
                */}
                {advisory.timings && (
                  <p className="advisory-timings">
                    {Object.entries(advisory.timings)
                      .filter(([nama]) => !nama.startsWith("_"))
                      .map(
                        ([nama, detik]) =>
                          `${nama} ${Number(detik).toFixed(1)} dtk`,
                      )
                      .join(" · ")}
                    {advisory.timings._total != null
                      ? ` · total ${Number(
                          advisory.timings._total,
                        ).toFixed(1)} dtk`
                      : ""}
                  </p>
                )}

              </div>


              {/* Sources */}

              {advisory.citations &&
                advisory.citations.length > 0 && (

                  <div className="advisory-section">

                    <button
                      type="button"
                      className="rujukan-toggle"
                      onClick={() =>
                        setRujukanTerbuka((v) => !v)
                      }
                      aria-expanded={rujukanTerbuka}
                    >
                      Sumber Rujukan ({advisory.citations.length} dokumen)
                      <span aria-hidden="true">
                        {rujukanTerbuka ? " ⌃" : " ⌄"}
                      </span>
                    </button>

                    {rujukanTerbuka && (
                      <div className="rujukan-panel">

                        {/*
                          Peringatan ini INTI kejujuran sistem: potongan ditelusur
                          untuk kondisi TERPREDIKSI, bukan kondisi terkini. Bila
                          prediksinya keliru, pedoman yang dirujuk ikut keliru
                          sasaran — dan dokter berhak tahu itu sebelum membacanya.
                        */}
                        <p className="rujukan-catatan">
                          Seluruh potongan di bawah ditelusur untuk{" "}
                          <strong>
                            kondisi terprediksi: {advisory.risk_level}
                          </strong>
                          , bukan untuk kondisi terkini
                          {result?.prediction?.current_glucose != null
                            ? ` (${result.prediction.current_glucose} mg/dL)`
                            : ""}
                          . Bila kondisi terprediksi keliru, pedoman yang dirujuk
                          pun keliru sasaran.
                        </p>

                        {/*
                          Keterangan cara menelusur. DUA KALI teks ini pernah
                          salah, dan keduanya karena jalur produksi berubah tanpa
                          teksnya ikut berubah:
                            1. menyebut fusi vektor + BM25 dengan RRF, padahal
                               produksi dibekukan ke BM25 murni sejak T14;
                            2. menyebut penyusunan ulang oleh cross-encoder,
                               padahal rag.reranker.enabled disetel false pada
                               24 Agustus 2026 (batas memori instans 512 MB).
                          Yang dibaca dokter harus menggambarkan apa yang BENAR-
                          BENAR berjalan. Bila reranker dihidupkan lagi, kalimat
                          di bawah HARUS ikut diubah.
                        */}
                        <p className="rujukan-metode">
                          Rujukan diperoleh dengan pencocokan istilah (BM25) atas
                          korpus pedoman, dan lima teratas menurut skor BM25
                          ditampilkan apa adanya. Penyusunan ulang oleh
                          cross-encoder tersedia pada sistem tetapi dimatikan
                          pada penerapan ini karena batas memori server, sehingga
                          urutan yang Anda lihat sepenuhnya berasal dari
                          pencocokan istilah. Skor BM25 tidak ditampilkan karena
                          nilainya tidak sebanding antar-pertanyaan dan mudah
                          disalahartikan sebagai derajat kebenaran klinis.
                        </p>

                        <label className="rujukan-sakelar">
                          <input
                            type="checkbox"
                            checked={kutipanUtuh}
                            onChange={(e) =>
                              setKutipanUtuh(e.target.checked)
                            }
                          />
                          Tampilkan potongan dokumen secara utuh
                        </label>

                        {advisory.citations.map(
                          (citation, index) => (

                            <div
                              className="rujukan-item"
                              ref={(el) => {
                                sumberRefs.current[index] = el;
                              }}
                              key={`${
                                citation.chunk_id ??
                                citation.kb_id ??
                                "source"
                              }-${index}`}
                            >

                              <p className="rujukan-judul">
                                #{citation.rank ?? index + 1} ·{" "}
                                {citation.title ??
                                  citation.source ??
                                  "Dokumen pedoman klinis"}
                              </p>

                              <p className="rujukan-meta">
                                {citation.source}
                                {citation.year ? ` (${citation.year})` : ""}
                                {citation.page ? ` · ${citation.page}` : ""}
                                {citation.nama_dokumen ? " · " : ""}
                                {citation.nama_dokumen && (
                                  <code>{citation.nama_dokumen}</code>
                                )}
                              </p>

                              {(citation.snippet ||
                                citation.teks_lengkap) && (
                                <blockquote className="rujukan-kutipan">
                                  {kutipanUtuh
                                    ? citation.teks_lengkap ??
                                      citation.snippet
                                    : citation.snippet ??
                                      citation.teks_lengkap}
                                </blockquote>
                              )}

                              {/*
                                Membedakan "potongan memang sependek itu" dari
                                "tampilannya yang memotong". Tanpa penanda ini,
                                keluhan teks terpotong tidak dapat ditelusuri
                                sebabnya.
                              */}
                              {!kutipanUtuh &&
                                citation.snippet_terpotong && (
                                  <p className="rujukan-penanda">
                                    Kutipan dipangkas untuk tampilan
                                    {citation.cara_potong === "batas_kalimat"
                                      ? " pada batas kalimat"
                                      : citation.cara_potong === "batas_kata"
                                        ? " pada batas kata"
                                        : ""}
                                    . Centang di atas untuk membaca utuh.
                                  </p>
                                )}

                              {citation.mulai_kalimat_utuh === false && (
                                <p className="rujukan-penanda">
                                  Potongan ini dimulai di tengah kalimat sejak
                                  proses pengindeksan, bukan karena tampilan.
                                </p>
                              )}

                            </div>

                          ),
                        )}

                      </div>
                    )}

                  </div>

                )}


              {/* Disclaimer */}

              <p className="advisory-disclaimer">

                Clinical decision support only.
                Final medical decisions require
                assessment by a qualified healthcare
                professional.

              </p>

            </div>

          ) : (

            <p className="advisory-empty">

              Run Clinical Assessment untuk
              mendapatkan rekomendasi klinis
              berdasarkan hasil prediksi.

            </p>

          )}

        </section>


        {/* =================================================
            ERROR
            ================================================= */}

        {error && (

          <section className="error-card">

            <p className="eyebrow">
              Error
            </p>

            <p>
              {error}
            </p>

            <button
              className="primary-button"
              onClick={handleRunClinical}
              disabled={loading}
            >
              Retry
            </button>

          </section>

        )}


        {/* =================================================
            ASSESSMENT HISTORY
            ================================================= */}

        <section className="section">

          <div className="section-heading">

            <div>

              <p className="eyebrow">
                History
              </p>

              <h2>
                Assessment history
              </h2>

            </div>

          </div>


          {history.length > 0 ? (

            <div className="history-list">

              {history
                .slice(0, 5)
                .map(
                  (item) => (

                    <article
                      className="history-card"
                      key={item.id}
                    >

                      <div>

                        <strong>

                          {item.current_glucose !==
                          null
                            ? item.current_glucose.toFixed(
                                1,
                              )
                            : "—"}

                          {(item.prediction_30m !== null ||
                            item.prediction_60m !== null) && (
                            <>
                              {" → "}

                              {item.prediction_30m !==
                              null
                                ? item.prediction_30m.toFixed(
                                    1,
                                  )
                                : "—"}

                              {" → "}

                              {item.prediction_60m !==
                              null
                                ? item.prediction_60m.toFixed(
                                    1,
                                  )
                                : "—"}
                            </>
                          )}

                          {" mg/dL"}

                        </strong>


                        <p className="muted">

                          {new Date(
                            item.assessed_at,
                          ).toLocaleString(
                            "id-ID",
                          )}

                        </p>

                      </div>


                      <div className="history-status">

                        <span
                          className={`risk-badge ${
                            item.condition
                              ?.toLowerCase() ===
                            "bahaya"
                              ? "danger"
                              : item.condition
                                    ?.toLowerCase() ===
                                "waspada"
                                ? "warning"
                                : "safe"
                          }`}
                        >

                          {item.glucose_source} · {item.condition
                            ?.toUpperCase() ||
                            "NORMAL"}

                        </span>


                        <span className="muted">
                          {item.prediction_30m !== null ||
                          item.prediction_60m !== null
                            ? item.grounded
                              ? "Prediction + grounded"
                              : "Prediction"
                            : "Current state"}
                        </span>

                      </div>

                    </article>

                  ),
                )}

            </div>

          ) : (

            <div className="history-empty">

              No assessment history available.

            </div>

          )}

        </section>

      </main>


      {/* ===================================================
          GLOBAL FOOTER
          =================================================== */}

      <Footer />

    </>
  );
}

// =============================================================
// CONTEXT CARD
// =============================================================

function ContextCard({
  title,
  value,
  unit,
}: {
  title: string;
  value: string;
  unit: string;
}) {
  return (
    <article className="context-card">

      <p className="label">
        {title}
      </p>

      <strong>
        {value}
      </strong>

      <span>
        {unit}
      </span>

    </article>
  );
}