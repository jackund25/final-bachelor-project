import { supabase } from "./supabase";
import { getPatientId } from "./glucose";
export async function saveAssessment({
  patientId,
  glucoseSourceId,
  currentGlucose,
  prediction30m,
  prediction60m,
  condition,
  recommendation,
  grounded,
  sources,
}: {
  patientId: string;
  glucoseSourceId: string;
  currentGlucose: number;
  prediction30m: number | null;
  prediction60m: number | null;
  condition: string | null;
  recommendation: string;
  grounded: boolean;
  sources: unknown[];
}) {
  const patientDbId = await getPatientId(patientId);
  if (!patientDbId) throw new Error("Patient tidak ditemukan.");

  const { data, error } = await supabase
    .from("clinical_assessments")
    .insert({
      patient_id: patientDbId,
      glucose_source_id: glucoseSourceId,
      current_glucose: currentGlucose,
      prediction_30m: prediction30m,
      prediction_60m: prediction60m,
      condition,
      recommendation,
      grounded,
      sources,
    })
    .select()
    .single();

  if (error) {
    throw new Error(
      `Failed to save assessment: ${error.message}`,
    );
  }

  return data;
}

export async function getAssessmentHistory(
  patientCode: string,
) {
  const patientDbId = await getPatientId(patientCode);
  if (!patientDbId) return [];

  const { data, error } = await supabase
    .from("clinical_assessments")
    .select(
      "id, patient_id, glucose_source_id, assessed_at, current_glucose, prediction_30m, prediction_60m, condition, recommendation, grounded, sources",
    )
    .eq("patient_id", patientDbId)
    .order("assessed_at", {
      ascending: false,
    })
    .limit(5);

  if (error) {
    console.error("[Supabase][clinical_assessments] history query error", {
      patientCode,
      patientId: patientDbId,
      error,
    });
    throw new Error(
      `Failed to load assessment history: ${error.message}`
    );
  }

  const sourceIds = Array.from(
    new Set(
      (data ?? [])
        .map((item) => item.glucose_source_id)
        .filter(Boolean),
    ),
  );

  const sourceTypes = new Map<string, string>();
  if (sourceIds.length > 0) {
    const { data: sources, error: sourceError } = await supabase
      .from("glucose_sources")
      .select("id, source_type")
      .in("id", sourceIds);

    if (sourceError) {
      console.error("[Supabase][glucose_sources] history source query error", {
        patientCode,
        sourceIds,
        error: sourceError,
      });
      throw new Error(
        `Failed to load assessment sources: ${sourceError.message}`,
      );
    }

    for (const source of sources ?? []) {
      sourceTypes.set(String(source.id), source.source_type);
    }
  }

  console.info("[Supabase][clinical_assessments] history query result", {
    patientCode,
    patientId: patientDbId,
    count: data?.length ?? 0,
  });

  return (data ?? []).map((item) => ({
    ...item,
    patient_id: patientCode,
    glucose_source:
      sourceTypes.get(String(item.glucose_source_id)) ?? null,
  }));
}