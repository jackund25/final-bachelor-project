"use client";

import {
  FormEvent,
  useEffect,
  useState,
} from "react";

import Navbar from "../../components/Navbar";
import Footer from "../../components/Footer";

import {
  getLogbookEntries,
  saveLogbookEntry,
  type LogbookEntry,
} from "../../lib/logbook";
import { getPatientSources } from "../../lib/glucose";
import type { GlucoseSource } from "../../lib/api";

import "../dashboard.css";
import "./logbook.css";

// =============================================================
// TYPES
// =============================================================

type LogbookRow = LogbookEntry & {
  id?: number | string;
};

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


// =============================================================
// VARIABLE GUIDE
// =============================================================

function VariableGuide() {
  return (
    <section className="logbook-card variable-guide-card">
      <details className="variable-guide">
        <summary className="variable-guide-summary">
          <div>
            <p className="eyebrow">REFERENCE</p>
            <h2>Arti dan Rentang Variabel</h2>
            <p className="variable-guide-subtitle">
              Panduan singkat untuk membantu memahami nilai yang dicatat.
            </p>
          </div>
          <span className="variable-guide-chevron" aria-hidden="true">
            ⌄
          </span>
        </summary>

        <div className="variable-guide-content">
          <div className="variable-guide-item">
            <h3>Glukosa — 40–400 mg/dL</h3>
            <ul>
              <li>Kadar glukosa darah saat pencatatan, yang menunjukkan kondisi glukosa pasien pada waktu tersebut.</li>
              <li><code>&lt;54 mg/dL</code> → hipoglikemia berat</li>
              <li><code>54–69 mg/dL</code> → hipoglikemia</li>
              <li><code>70–180 mg/dL</code> → rentang sasaran</li>
              <li><code>181–250 mg/dL</code> → hiperglikemia</li>
              <li><code>&gt;250 mg/dL</code> → hiperglikemia berat</li>
            </ul>
          </div>

          <div className="variable-guide-item">
            <h3>Karbohidrat — 0–200 g</h3>
            <ul>
              <li>Jumlah karbohidrat dalam satu kali asupan, bukan total konsumsi harian. Nilai ini membantu menggambarkan asupan yang dapat memengaruhi kadar glukosa setelah makan.</li>
              <li>±15 g → sepotong roti</li>
              <li>±25 g → satu buah pisang</li>
              <li>±40 g → satu porsi nasi</li>
              <li>Pengaruh asupan terhadap glukosa berlangsung secara bertahap selama sekitar 3 jam.</li>
            </ul>
          </div>

          <div className="variable-guide-item">
            <h3>Insulin — 0–30 unit</h3>
            <ul>
              <li>Dosis insulin bolus yang diberikan pada saat pencatatan, bukan insulin basal. Nilai ini menggambarkan insulin yang diberikan untuk membantu mengendalikan kadar glukosa.</li>
              <li>±2 unit → dosis bolus kecil</li>
              <li>±4–6 unit → dosis bolus sedang</li>
              <li>±10 unit → dosis bolus lebih tinggi</li>
              <li>Efek insulin berlangsung secara bertahap selama beberapa jam setelah pemberian.</li>
            </ul>
          </div>

          <div className="variable-guide-item">
            <h3>Aktivitas — 0–10</h3>
            <ul>
              <li>Menunjukkan intensitas aktivitas fisik, bukan durasi. Semakin tinggi nilai, semakin berat aktivitas yang dilakukan.</li>
              <li><code>0</code> → tidak beraktivitas</li>
              <li><code>1–3</code> → aktivitas ringan, seperti jalan santai atau pekerjaan rumah</li>
              <li><code>4–6</code> → aktivitas sedang, seperti jalan cepat atau bersepeda santai</li>
              <li><code>7–8</code> → aktivitas berat, seperti lari atau berenang</li>
              <li><code>9–10</code> → aktivitas sangat berat, seperti lari cepat atau angkat beban</li>
              <li className="variable-guide-catatan">
                Dicatat sebagai konteks klinis. Pada data pelatihan, kontribusi
                aktivitas terhadap prediksi glukosa terukur sangat kecil (0,18%),
                jadi mengisi nilai ini hampir tidak menggeser angka prediksi.
              </li>
            </ul>
          </div>

        </div>
      </details>
    </section>
  );
}

// =============================================================
// PAGE
// =============================================================

export default function LogbookPage() {
  // ===========================================================
  // PATIENT
  // ===========================================================

  const [patientId, setPatientId] = useState("");
  const [glucoseSource, setGlucoseSource] =
    useState<GlucoseSource>("CGM");
  const [sourceOptions, setSourceOptions] =
    useState<GlucoseSource[]>([]);

  // ===========================================================
  // FORM STATE
  // ===========================================================

  const [timestamp, setTimestamp] =
    useState("");

  const [glucose, setGlucose] =
    useState("");

  const [insulin, setInsulin] =
    useState("");

  const [carbs, setCarbs] =
    useState("");

  const [activity, setActivity] =
    useState("");

  // ===========================================================
  // DATA STATE
  // ===========================================================

  const [entries, setEntries] =
    useState<LogbookRow[]>([]);

  const [loading, setLoading] =
    useState(false);

  const [loadingHistory, setLoadingHistory] =
    useState(true);

  const [message, setMessage] =
    useState<string | null>(null);

  const [error, setError] =
    useState<string | null>(null);

  // ===========================================================
  // SET DEFAULT DATE/TIME
  // ===========================================================

  useEffect(() => {
    const now = new Date();

    const localDateTime =
      new Date(
        now.getTime() -
          now.getTimezoneOffset() *
            60000,
      )
        .toISOString()
        .slice(0, 16);

    setTimestamp(localDateTime);
  }, []);

  // ===========================================================
  // LOAD LOGBOOK
  // ===========================================================

  const loadEntries = async () => {
    try {
      setLoadingHistory(true);
      setError(null);

      const data =
        await getLogbookEntries(
          patientId,
          glucoseSource,
        );

      setEntries(
        (data ?? []) as LogbookRow[],
      );
    } catch (err) {
      console.error(
        "Failed to load logbook:",
        err,
      );

      setError(
        err instanceof Error
          ? err.message
          : "Failed to load logbook.",
      );
    } finally {
      setLoadingHistory(false);
    }
  };

  // ===========================================================
  // LOAD DATA ON PAGE OPEN
  // ===========================================================

  useEffect(() => {
    const storedPatient = window.localStorage.getItem(
      "last_patient_id",
    );
    const storedSource = window.localStorage.getItem(
      "last_glucose_source",
    );

    setPatientId(storedPatient ?? "");
    if (
      storedSource === "CGM" ||
      storedSource === "FINGER_STICK"
    ) {
      setGlucoseSource(storedSource);
    }
  }, []);

  useEffect(() => {
    if (!patientId) {
      return;
    }
    getPatientSources(patientId).then((sources) => {
      const available = sources.map((item) => item.source_type);
      setSourceOptions(available);
      if (!available.includes(glucoseSource) && available[0]) {
        setGlucoseSource(available[0]);
      }
    }).catch((loadError) => {
      console.error("Failed to load glucose sources:", loadError);
      setSourceOptions([]);
    });
    window.localStorage.setItem(
      "last_patient_id",
      patientId,
    );
    window.localStorage.setItem(
      "last_glucose_source",
      glucoseSource,
    );
    loadEntries();
  }, [patientId, glucoseSource]);

  // ===========================================================
  // SUBMIT LOGBOOK
  // ===========================================================

  const handleSubmit = async (
    event: FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault();

    setLoading(true);
    setMessage(null);
    setError(null);

    try {
      // -------------------------------------------------------
      // VALIDATION
      // -------------------------------------------------------

      if (!timestamp) {
        throw new Error(
          "Please select the date and time.",
        );
      }

      if (!glucose) {
        throw new Error(
          "Please enter glucose.",
        );
      }

      if (!insulin) {
        throw new Error(
          "Please enter insulin.",
        );
      }

      if (!carbs) {
        throw new Error(
          "Please enter carbohydrate.",
        );
      }

      if (!activity) {
        throw new Error(
          "Please enter activity.",
        );
      }

      // -------------------------------------------------------
      // CREATE LOGBOOK ENTRY
      // -------------------------------------------------------

      const entry: LogbookEntry = {
        patient_id: patientId,

        timestamp:
          new Date(
            timestamp,
          ).toISOString(),

        glucose: Number(glucose),

        insulin: Number(insulin),

        carbs: Number(carbs),

        activity: Number(activity),

        glucose_source: glucoseSource,
      };

      // -------------------------------------------------------
      // SAVE TO SUPABASE
      // -------------------------------------------------------

      await saveLogbookEntry(
        entry,
      );

      // -------------------------------------------------------
      // SUCCESS
      // -------------------------------------------------------

      setMessage(
        "Logbook entry saved successfully.",
      );

      // -------------------------------------------------------
      // RESET FORM
      // -------------------------------------------------------

      setGlucose("");
      setInsulin("");
      setCarbs("");
      setActivity("");

      // -------------------------------------------------------
      // RELOAD HISTORY
      // -------------------------------------------------------

      await loadEntries();

    } catch (err) {
      console.error(
        "Failed to save logbook:",
        err,
      );

      setError(
        err instanceof Error
          ? err.message
          : "Failed to save logbook.",
      );
    } finally {
      setLoading(false);
    }
  };

  // ===========================================================
  // RENDER
  // ===========================================================

  return (
    <>
      {/* =====================================================
          NAVBAR
          ===================================================== */}

      <Navbar />

      {/* =====================================================
          MAIN
          ===================================================== */}

      <main className="logbook-page">

        {/* ===================================================
            PAGE HEADER
            =================================================== */}

        <header className="logbook-header">

          <div>

            <p className="eyebrow">
              PATIENT LOGBOOK
            </p>

            <h1>
              Patient state
            </h1>

            <p className="logbook-description">
              Record the patient's recent
              glucose and contextual state.
            </p>

          </div>

          <div className="patient-badge">

            <span className="status-dot" />

            Patient {patientId}

          </div>

        </header>


        {/* ===================================================
            VARIABLE GUIDE
            =================================================== */}

        <VariableGuide />


        {/* ===================================================
            INPUT CARD
            =================================================== */}

        <section className="logbook-card">

          <div className="section-heading">

            <div>

              <p className="eyebrow">
                NEW ENTRY
              </p>

              <h2>
                Record patient state
              </h2>

            </div>

          </div>


          {/* =================================================
              FORM
              ================================================= */}

          <form
            className="logbook-form"
            onSubmit={handleSubmit}
          >

            {/* DATE & TIME */}

            <div className="form-field full">

                <label htmlFor="glucose-source">
                  Glucose source
                </label>

                <select
                  id="glucose-source"
                  value={glucoseSource}
                  onChange={(event) =>
                    setGlucoseSource(
                      event.target.value as GlucoseSource,
                    )
                  }
                >
                  {sourceOptions.map((source) => (
                    <option key={source} value={source}>
                      {source === "CGM" ? "CGM" : "Finger-stick / SMBG"}
                    </option>
                  ))}
                </select>

              </div>

            <div className="form-field full">

                <label htmlFor="timestamp">
                Date & time
              </label>

              <input
                id="timestamp"
                type="datetime-local"
                value={timestamp}
                onChange={(event) =>
                  setTimestamp(
                    event.target.value,
                  )
                }
                required
              />

            </div>


            {/* GLUCOSE */}

            <div className="form-field">

              <label htmlFor="glucose">
                Glucose
              </label>

              <div className="input-with-unit">

                <input
                  id="glucose"
                  type="number"
                  min="40"
                  max="400"
                  step="0.1"
                  placeholder="125"
                  value={glucose}
                  onChange={(event) =>
                    setGlucose(
                      event.target.value,
                    )
                  }
                  required
                />

                <span>
                  mg/dL
                </span>

              </div>

            </div>


            {/* INSULIN */}

            <div className="form-field">

              <label htmlFor="insulin">
                Insulin
              </label>

              <div className="input-with-unit">

                <input
                  id="insulin"
                  type="number"
                  min="0"
                  max="30"
                  step="0.1"
                  placeholder="2"
                  value={insulin}
                  onChange={(event) =>
                    setInsulin(
                      event.target.value,
                    )
                  }
                  required
                />

                <span>
                  units
                </span>

              </div>

            </div>


            {/* CARBOHYDRATE */}

            <div className="form-field">

              <label htmlFor="carbs">
                Carbohydrate
              </label>

              <div className="input-with-unit">

                <input
                  id="carbs"
                  type="number"
                  min="0"
                  max="200"
                  step="1"
                  placeholder="30"
                  value={carbs}
                  onChange={(event) =>
                    setCarbs(
                      event.target.value,
                    )
                  }
                  required
                />

                <span>
                  g
                </span>

              </div>

            </div>


            {/* ACTIVITY */}

            <div className="form-field">

              <label htmlFor="activity">
                Activity
              </label>

              <div className="input-with-unit">

                <input
                  id="activity"
                  type="number"
                  min="0"
                  max="10"
                  step="1"
                  placeholder="3"
                  value={activity}
                  onChange={(event) =>
                    setActivity(
                      event.target.value,
                    )
                  }
                  required
                />

                <span>
                  level
                </span>

              </div>

            </div>



            {/* =================================================
                MESSAGE
                ================================================= */}

            {message && (
              <div className="form-message success">
                {message}
              </div>
            )}

            {error && (
              <div className="form-message error">
                {error}
              </div>
            )}


            {/* =================================================
                SUBMIT BUTTON
                ================================================= */}

            <div className="form-actions">

              <button
                type="submit"
                className="primary-button"
                disabled={loading}
              >
                {loading
                  ? "Saving..."
                  : "Save logbook entry"}
              </button>

            </div>

          </form>

        </section>



        {/* ===================================================
            CURRENT PATIENT STATE
            =================================================== */}

        <section className="logbook-card">

          <div className="section-heading">

            <div>

              <p className="eyebrow">
                CURRENT INPUT
              </p>

              <h2>
                Patient state preview
              </h2>

            </div>

          </div>


          <div className="context-grid">

            <ContextCard
              title="Glucose"
              value={
                glucose || "—"
              }
              unit="mg/dL"
            />

            <ContextCard
              title="Insulin"
              value={
                insulin || "—"
              }
              unit="units"
            />

            <ContextCard
              title="Carbohydrate"
              value={
                carbs || "—"
              }
              unit="g"
            />

            <ContextCard
              title="Activity"
              value={
                activity || "—"
              }
              unit="level"
            />

          </div>

        </section>


        {/* ===================================================
            RECENT LOGBOOK HISTORY
            =================================================== */}

        <section className="logbook-card">

          <div className="section-heading">

            <div>

              <p className="eyebrow">
                HISTORY
              </p>

              <h2>
                Recent logbook entries
              </h2>

            </div>

            <span className="muted">
              Last {entries.length}
            </span>

          </div>


          {loadingHistory ? (

            <p className="empty-state">
              Loading logbook...
            </p>

          ) : entries.length === 0 ? (

            <p className="empty-state">
              No logbook entries yet.
            </p>

          ) : (

            <div className="logbook-table-wrapper">

              <table className="logbook-table">

                <thead>

                  <tr>

                    <th>
                      Date & time
                    </th>

                    <th>
                      Glucose
                    </th>

                    <th>
                      Insulin
                    </th>

                    <th>
                      Carbs
                    </th>

                    <th>
                      Activity
                    </th>

                  </tr>

                </thead>

                <tbody>

                  {entries.map(
                    (entry) => (

                      <tr
                        key={
                          entry.id != null
                            ? `${entry.glucose_source}-${entry.id}`
                            : `${entry.glucose_source}-${entry.timestamp}-${entry.glucose}`
                        }
                      >

                        <td>
                          {new Date(
                            entry.timestamp,
                          ).toLocaleString(
                            "id-ID",
                            {
                              dateStyle:
                                "short",

                              timeStyle:
                                "short",
                            },
                          )}
                        </td>

                        <td>
                          <strong>
                            {entry.glucose}
                          </strong>{" "}
                          mg/dL
                        </td>

                        <td>
                          {entry.insulin}{" "}
                          units
                        </td>

                        <td>
                          {entry.carbs} g
                        </td>

                        <td>
                          {entry.activity}
                        </td>

                      </tr>

                    ),
                  )}

                </tbody>

              </table>

            </div>

          )}

        </section>

      </main>


      {/* =====================================================
          FOOTER
          ===================================================== */}

      <Footer />
    </>
  );
}