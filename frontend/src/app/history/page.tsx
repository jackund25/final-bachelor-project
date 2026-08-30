"use client";

import { useEffect, useState } from "react";
import "../dashboard.css";
import "./history.css";

import Navbar from "../../components/Navbar";
import Footer from "../../components/Footer";
import { getAssessmentHistory } from "../../lib/assessment";
import { getPatients } from "../../lib/glucose";

type HistoryItem = {
  id: number | string;
  assessed_at: string;
  current_glucose: number | null;
  prediction_30m: number | null;
  prediction_60m: number | null;
  condition: string | null;
  glucose_source: string | null;
  recommendation: string | null;
  grounded: boolean;
  sources: Array<{
    source?: string;
    title?: string;
    year?: number | string;
    page?: number | string;
    kb_id?: string;
  }>;
};

export default function HistoryPage() {
  const [patientId, setPatientId] = useState("");
  const [patients, setPatients] = useState<string[]>([]);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = async (patientCode: string) => {
    if (!patientCode) {
      setHistory([]);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await getAssessmentHistory(patientCode);
      setHistory((data ?? []) as HistoryItem[]);
    } catch (loadError) {
      console.error("Failed to load assessment history:", loadError);
      setError("Assessment history gagal dimuat.");
      setHistory([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const loadPatients = async () => {
      try {
        const records = await getPatients();
        const availablePatients = records.map((item) => item.patient_code);
        const storedPatient = window.localStorage.getItem("last_patient_id");
        const selectedPatient = storedPatient && availablePatients.includes(storedPatient)
          ? storedPatient
          : availablePatients[0] ?? "";

        setPatients(availablePatients);
        setPatientId(selectedPatient);
      } catch (loadError) {
        console.error("Failed to load patients:", loadError);
        setError("Daftar patient gagal dimuat.");
        setLoading(false);
      }
    };

    loadPatients();
  }, []);

  useEffect(() => {
    if (!patientId) return;

    window.localStorage.setItem("last_patient_id", patientId);
    loadHistory(patientId);
  }, [patientId]);

  return (
    <>
      <Navbar />

      <main className="history-page">
        <header className="history-header">
          <div>
            <p className="eyebrow">ASSESSMENT HISTORY</p>
            <h1>Clinical assessments</h1>
            <p className="history-description">
              Review the latest clinical decision-support assessments for the selected patient.
            </p>
          </div>

          <label className="history-patient-selector">
            <span>Patient</span>
            <select
              value={patientId}
              onChange={(event) => setPatientId(event.target.value)}
              disabled={patients.length === 0}
            >
              {patients.map((patient) => (
                <option key={patient} value={patient}>
                  {patient}
                </option>
              ))}
            </select>
          </label>
        </header>

        {error && <div className="history-message error">{error}</div>}

        <section className="history-panel">
          <div className="history-panel-heading">
            <div>
              <p className="eyebrow">PATIENT RECORD</p>
              <h2>{patientId || "No patient selected"}</h2>
            </div>
            <span className="muted">Latest 5 assessments</span>
          </div>

          {loading ? (
            <p className="history-empty">Loading assessment history...</p>
          ) : history.length === 0 ? (
            <p className="history-empty">No assessment history available for this patient.</p>
          ) : (
            <div className="history-items">
              {history.slice(0, 5).map((item) => (
                <article className="history-item" key={item.id}>
                  <div className="history-item-topline">
                    <div>
                      <span className="history-date">
                        {new Date(item.assessed_at).toLocaleString("id-ID")}
                      </span>
                      <h3>{item.condition ?? "Condition unavailable"}</h3>
                    </div>
                    <span className="history-source">
                      {item.glucose_source ?? "Source unavailable"}
                    </span>
                  </div>

                  <div className="history-values">
                    <span>Current <strong>{item.current_glucose?.toFixed(1) ?? "—"}</strong> mg/dL</span>
                    <span>30m <strong>{item.prediction_30m?.toFixed(1) ?? "—"}</strong> mg/dL</span>
                    <span>60m <strong>{item.prediction_60m?.toFixed(1) ?? "—"}</strong> mg/dL</span>
                  </div>

                  {item.recommendation && (
                    <p className="history-recommendation">{item.recommendation}</p>
                  )}

                  <div className="history-item-footer">
                    <span className={item.grounded ? "history-grounded" : "muted"}>
                      {item.grounded ? "Grounded recommendation" : "Not grounded"}
                    </span>
                    <span className="muted">
                      {item.sources?.length ?? 0} evidence source{item.sources?.length === 1 ? "" : "s"}
                    </span>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </main>

      <Footer />
    </>
  );
}
