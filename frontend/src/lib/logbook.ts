import { API_BASE_URL, headerApi, type GlucoseSource } from "./api";
import { getGlucoseObservations } from "./glucose";

export type LogbookEntry = {
  id?: number;
  patient_id: string;
  timestamp: string;
  glucose: number;
  insulin: number;
  carbs: number;
  activity: number;
  glucose_source: GlucoseSource;
};

export async function saveLogbookEntry(
  entry: LogbookEntry,
) {
  // Memakai API_BASE_URL bersama, bukan menyalin ulang logikanya. Salinan yang
  // dulu ada di sini memakai `||`, sehingga mode satu-asal (string kosong) jatuh
  // kembali ke 127.0.0.1 — dan penyimpanan logbook akan menembak localhost dari
  // peramban dokter, yang di sana berarti mesin DOKTER, bukan server.
  const response = await fetch(`${API_BASE_URL}/api/logbook`, {
    method: "POST",
    headers: headerApi({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      patient_code: entry.patient_id,
      glucose_source: entry.glucose_source,
      timestamp: entry.timestamp,
      glucose: entry.glucose,
      insulin: entry.insulin,
      carbs: entry.carbs,
      activity: entry.activity,
    }),
  });

  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.detail ?? "Failed to save logbook entry.");
  }

  return result.data;
}

export async function getLogbookEntries(
  patientId: string,
  glucoseSource?: GlucoseSource,
) {
  if (!glucoseSource) return [];

  const observations = await getGlucoseObservations(
    patientId,
    glucoseSource,
  );

  return observations.slice(-20).reverse();
}