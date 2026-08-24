import { supabase } from "./supabase";
import type { GlucoseSource, Observation } from "./api";

export type GlucosePoint = {
  timestamp: string;
  glucose: number;
};

export type PatientOption = {
  id: string;
  patient_code: string;
};

export type GlucoseSourceOption = {
  id: string;
  source_type: GlucoseSource;
};

export type PatientContext = {
  insulin: number;
  carbs: number;
  activity: number;
};

export async function getPatients(): Promise<PatientOption[]> {
  const { data, error } = await supabase
    .from("patients")
    .select("id, patient_code")
    .order("patient_code", { ascending: true });

  if (error) {
    throw new Error(`Failed to load patients: ${error.message}`);
  }

  if (!data?.length) {
    console.warn("[Supabase][patients] query berhasil tetapi 0 rows", {
      table: "patients",
    });
  }

  return (data ?? []) as PatientOption[];
}

export async function getPatientSources(
  patientCode: string,
): Promise<GlucoseSourceOption[]> {
  const patient = await getPatientByCode(patientCode);
  if (!patient) return [];

  const { data, error } = await supabase
    .from("glucose_sources")
    .select("id, source_type")
    .eq("patient_id", patient.id)
    .order("source_type", { ascending: true });

  if (error) {
    console.error("[Supabase][glucose_sources] query error", {
      patientCode,
      patientId: patient.id,
      error,
    });
    throw new Error(`Failed to load glucose sources: ${error.message}`);
  }

  console.info("[Supabase][glucose_sources] query result", {
    patientCode,
    patientId: patient.id,
    count: data?.length ?? 0,
  });

  return (data ?? []) as GlucoseSourceOption[];
}

async function getPatientByCode(
  patientCode: string,
): Promise<PatientOption | null> {
  const { data, error } = await supabase
    .from("patients")
    .select("id, patient_code")
    .eq("patient_code", patientCode)
    .maybeSingle();

  if (error) {
    console.error("[Supabase][patients] patient lookup error", {
      patientCode,
      error,
    });
    throw new Error(`Failed to resolve patient: ${error.message}`);
  }

  return data as PatientOption | null;
}

export async function getPatientId(
  patientCode: string,
): Promise<string | null> {
  const patient = await getPatientByCode(patientCode);
  return patient?.id ?? null;
}

async function getSourceId(
  patientId: string,
  glucoseSource: GlucoseSource,
): Promise<string | null> {
  const { data, error } = await supabase
    .from("glucose_sources")
    .select("id")
    .eq("patient_id", patientId)
    .eq("source_type", glucoseSource)
    .maybeSingle();

  if (error) {
    console.error("[Supabase][glucose_sources] source lookup error", {
      patientId,
      glucoseSource,
      error,
    });
    throw new Error(`Failed to resolve glucose source: ${error.message}`);
  }

  return data?.id ?? null;
}

export async function getGlucoseObservations(
  patientCode: string,
  glucoseSource: GlucoseSource,
): Promise<Observation[]> {
  const patient = await getPatientByCode(patientCode);
  if (!patient) return [];

  const activePatientId = Number(patient.id);
  if (!Number.isInteger(activePatientId)) {
    throw new Error(`Invalid internal patient ID for ${patientCode}.`);
  }

  const sourceId = await getSourceId(patient.id, glucoseSource);
  if (!sourceId) {
    console.warn("[Supabase][glucose_sources] source tidak tersedia", {
      patientCode,
      glucoseSource,
    });
    return [];
  }

  const [readings, insulin, meals, activity] = await Promise.all([
    supabase
      .from("glucose_readings")
      .select("id, timestamp, glucose")
      .eq("glucose_source_id", sourceId)
      .order("timestamp", { ascending: true }),
    supabase
      .from("insulin_events")
      .select("timestamp, insulin_units")
      .eq("patient_id", activePatientId)
      .order("timestamp", { ascending: true }),
    supabase
      .from("meal_events")
      .select("timestamp, carbs_grams")
      .eq("patient_id", activePatientId)
      .order("timestamp", { ascending: true }),
    supabase
      .from("activity_events")
      .select("timestamp, activity_level")
      .eq("patient_id", activePatientId)
      .order("timestamp", { ascending: true }),
  ]);

  const firstError = [
    readings.error,
    insulin.error,
    meals.error,
    activity.error,
  ].find(Boolean);

  const validReadingCount = (readings.data ?? []).filter(
    (reading) => Number.isFinite(Number(reading.glucose)),
  ).length;
  const minimumRequired = glucoseSource === "FINGER_STICK" ? 8 : 12;

  console.info("[Supabase][availability] readings", {
    patientCode,
    source: glucoseSource,
    sourceId,
    rawReadingCount: readings.data?.length ?? 0,
    validReadingCount,
    minimumRequired,
    hasSufficientHistory: validReadingCount >= minimumRequired,
    predictionArtifactAvailable: true,
  });

  if (firstError) {
    console.error("[Supabase][patient data] query error", {
      patientCode,
      glucoseSource,
      sourceId,
      errors: [
        ["glucose_readings", readings.error],
        ["insulin_events", insulin.error],
        ["meal_events", meals.error],
        ["activity_events", activity.error],
      ].filter(([, queryError]) => queryError),
    });
    throw new Error(`Failed to load patient data: ${firstError.message}`);
  }

  console.info("[Supabase][patient data] query result", {
    patientCode,
    glucoseSource,
    sourceId,
    readings: readings.data?.length ?? 0,
    insulinEvents: insulin.data?.length ?? 0,
    mealEvents: meals.data?.length ?? 0,
    activityEvents: activity.data?.length ?? 0,
  });

  const latestBefore = <T extends { timestamp: string }>(
    events: T[],
    timestamp: string,
  ) => events.reduce<T | undefined>(
    (latest, event) => event.timestamp <= timestamp ? event : latest,
    undefined,
  );

  const insulinEvents = insulin.data ?? [];
  const mealEvents = meals.data ?? [];
  const activityEvents = activity.data ?? [];

  return (readings.data ?? []).map((reading) => {
    const insulinEvent = latestBefore(insulinEvents, reading.timestamp);
    const mealEvent = latestBefore(mealEvents, reading.timestamp);
    const activityEvent = latestBefore(activityEvents, reading.timestamp);

    return {
      id: reading.id,
      patient_id: patientCode,
      timestamp: reading.timestamp,
      glucose: Number(reading.glucose),
      insulin: Number(insulinEvent?.insulin_units ?? 0),
      carbs: Number(mealEvent?.carbs_grams ?? 0),
      activity: Number(activityEvent?.activity_level ?? 0),
      glucose_source: glucoseSource,
    };
  });
}

export async function getGlucoseHistory(
  patientCode: string,
  glucoseSource: GlucoseSource,
  limit = 24,
): Promise<GlucosePoint[]> {
  const observations = await getGlucoseObservations(patientCode, glucoseSource);
  return observations.slice(-limit).map((item) => ({
    timestamp: item.timestamp,
    glucose: item.glucose,
  }));
}

export async function getGlucoseSourceId(
  patientCode: string,
  glucoseSource: GlucoseSource,
): Promise<string | null> {
  const patient = await getPatientByCode(patientCode);
  return patient
    ? getSourceId(patient.id, glucoseSource)
    : null;
}

export async function getLatestPatientContext(
  patientCode: string,
): Promise<PatientContext> {
  const patient = await getPatientByCode(patientCode);
  if (!patient) {
    return { insulin: 0, carbs: 0, activity: 0 };
  }

  const activePatientId = Number(patient.id);
  if (!Number.isInteger(activePatientId)) {
    throw new Error(`Invalid internal patient ID for ${patientCode}.`);
  }

  const [insulin, meals, activity] = await Promise.all([
    supabase
      .from("insulin_events")
      .select("timestamp, insulin_units")
      .eq("patient_id", activePatientId)
      .order("timestamp", { ascending: false })
      .limit(1)
      .maybeSingle(),
    supabase
      .from("meal_events")
      .select("timestamp, carbs_grams")
      .eq("patient_id", activePatientId)
      .order("timestamp", { ascending: false })
      .limit(1)
      .maybeSingle(),
    supabase
      .from("activity_events")
      .select("timestamp, activity_level")
      .eq("patient_id", activePatientId)
      .order("timestamp", { ascending: false })
      .limit(1)
      .maybeSingle(),
  ]);

  const firstError = [
    insulin.error,
    meals.error,
    activity.error,
  ].find(Boolean);

  if (firstError) {
    console.error("[Supabase][recent context] query error", {
      patientCode,
      patientId: activePatientId,
      error: firstError,
    });
    throw new Error(`Failed to load recent patient context: ${firstError.message}`);
  }

  const context = {
    insulin: Number(insulin.data?.insulin_units ?? 0),
    carbs: Number(meals.data?.carbs_grams ?? 0),
    activity: Number(activity.data?.activity_level ?? 0),
  };

  console.info("[Supabase][recent context] latest events", {
    patientCode,
    patientId: activePatientId,
    filters: {
      insulin_events: `patient_id = ${activePatientId}`,
      meal_events: `patient_id = ${activePatientId}`,
      activity_events: `patient_id = ${activePatientId}`,
    },
    rowCounts: {
      insulin_events: insulin.data ? 1 : 0,
      meal_events: meals.data ? 1 : 0,
      activity_events: activity.data ? 1 : 0,
    },
    insulin: { value: context.insulin, timestamp: insulin.data?.timestamp ?? null },
    carbs: { value: context.carbs, timestamp: meals.data?.timestamp ?? null },
    activity: { value: context.activity, timestamp: activity.data?.timestamp ?? null },
  });

  return context;
}
