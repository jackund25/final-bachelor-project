"use client";

import { useEffect, useState } from "react";
import "./dashboard.css";

import {
  runClinicalAssessment,
  type Observation,
  type ClinicalResponse,
  type GlucoseSource,
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
  citations: Citation[];
};

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
    useState({ insulin: 0, carbs: 0, activity: 0, stress: 0 });

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

  const stress = recentContext.stress;

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

  const predictionAvailable =
    result?.prediction?.prediction_available ??
    (prediction30 !== null || prediction60 !== null);

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
    result?.prediction?.minimum_required ?? 12;
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
    setRecentContext({ insulin: 0, carbs: 0, activity: 0, stress: 0 });
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
          predictionAvailable
            ? "Berikan rekomendasi klinis berdasarkan hasil prediksi glukosa pasien."
            : "Bagaimana kondisi pasien saat ini berdasarkan glukosa terakhir?",
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

          citations:
            result.clinical_advisory.citations ?? [],
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
                  `Insufficient historical ${glucoseSource} observations: ${observations.length}/12.`}
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
                        30 MIN
                      </span>

                      <strong>
                        {prediction30 !== null
                          ? prediction30.toFixed(1)
                          : "—"}
                      </strong>

                      <span>
                        mg/dL
                      </span>

                    </div>

                  </div>


                  <div className="forecast-line" />


                  {/* 60 MIN */}

                  <div className="forecast-point">

                    <div className="forecast-dot" />

                    <div className="forecast-info">

                      <span className="forecast-time">
                        60 MIN
                      </span>

                      <strong>
                        {prediction60 !== null
                          ? prediction60.toFixed(1)
                          : "—"}
                      </strong>

                      <span>
                        mg/dL
                      </span>

                    </div>

                  </div>

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
                      30-minute forecast
                    </span>

                    <strong>
                      {prediction30 !== null
                        ? `${prediction30.toFixed(1)} mg/dL`
                        : "Not available"}
                    </strong>

                  </div>


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

            <ContextCard
              title="Stress"
              value={String(
                stress,
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

                <p className="advisory-text">
                  {advisory.explanation}
                </p>

              </div>


              {/* Evidence */}

              <div className="advisory-section">

                <p className="advisory-label">
                  Evidence status
                </p>

                <p className="advisory-text">

                  {advisory.grounded
                    ? "Recommendation is grounded in the retrieved clinical knowledge base."
                    : "Retrieved evidence is insufficient for a grounded recommendation."}

                </p>

              </div>


              {/* Sources */}

              {advisory.citations &&
                advisory.citations.length >
                  0 && (

                  <div className="advisory-section">

                    <p className="advisory-label">
                      Sources
                    </p>


                    <div className="source-list">

                      {advisory.citations.map(
                        (
                          citation,
                          index,
                        ) => (

                          <span
                            className="source-chip"
                            key={`${
                              citation.kb_id ??
                              citation.source ??
                              "source"
                            }-${index}`}
                          >

                            {citation.source ?? citation.title ?? "Clinical knowledge source"}

                            {citation.year
                              ? ` · ${citation.year}`
                              : ""}

                            {citation.page
                              ? ` · p. ${citation.page}`
                              : ""}

                          </span>

                        ),
                      )}

                    </div>

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