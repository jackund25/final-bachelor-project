import type { GlucoseSource } from "./api";
import { getGlucoseObservations } from "./glucose";

export type LogbookEntry = {
  id?: number;
  patient_id: string;
  timestamp: string;
  glucose: number;
  insulin: number;
  carbs: number;
  activity: number;
  stress: number;
  glucose_source: GlucoseSource;
};

export async function saveLogbookEntry(
  entry: LogbookEntry,
) {
  const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}/api/logbook`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      patient_code: entry.patient_id,
      glucose_source: entry.glucose_source,
      timestamp: entry.timestamp,
      glucose: entry.glucose,
      insulin: entry.insulin,
      carbs: entry.carbs,
      activity: entry.activity,
      stress: entry.stress,
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