const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export type GlucoseSource = "CGM" | "FINGER_STICK";

export interface Observation {
  id?: number | string;
  patient_id: string;
  timestamp: string;
  glucose: number;
  insulin: number;
  carbs: number;
  activity: number;
  stress: number;
  glucose_source?: GlucoseSource;
}

export interface PredictionHorizon {
  horizon_minutes: number;
  prediction: number;
  std: number | null;
  interval: [number, number] | null;
  coverage: number | null;
  model_family: string;
}

export interface PredictionResult {
  mode?: "current_state" | "prediction";
  prediction_available: boolean;
  glucose_source: GlucoseSource;
  current_glucose: number | null;
  history?: {
    available: boolean;
    history_observations: number;
    history_required: number;
    history_required_message?: string;
    reason?: string;
  };
  horizons: PredictionHorizon[];
  prediction?: number | null;
  prediction_30m?: number | null;
  prediction_60m?: number | null;
  condition?: string | null;
  reason?: string | null;
  prediction_artifact_available?: boolean;
  minimum_required?: number;
  raw_reading_count?: number;
  valid_reading_count?: number;
  has_sufficient_history?: boolean;
}

export interface Citation {
  source?: string;
  title?: string;
  year?: number;
  page?: number;
  kb_id?: string;
}

export interface ClinicalAdvisory {
  mode?: "current_state" | "prediction_conditioned";
  explanation: string;
  risk_level: string;
  grounded: boolean;
  citations: Citation[];
  retrieved_docs: unknown[];
  recommendation_available?: boolean;
  prediction_horizon_minutes?: number;
  prediction?: number | null;
  disclaimer?: string;
  source?: GlucoseSource;
  reason?: string;
}

export interface ClinicalResponse {
  status: string;
  patient_code: string;
  glucose_source: GlucoseSource;
  mode: "current_state" | "prediction";
  prediction: PredictionResult;
  clinical_advisory: ClinicalAdvisory;
}

export async function runClinicalAssessment(
  patientId: string,
  question: string,
  observations: Observation[],
  glucoseSource: GlucoseSource = "CGM",
): Promise<ClinicalResponse> {
  const response = await fetch(`${API_BASE_URL}/api/clinical`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      patient_code: patientId,
      glucose_source: glucoseSource,
      question,
      data: observations.map((item) => ({
        ...item,
        glucose_source: glucoseSource,
      })),
    }),
  });

  const result = await response.json();

  if (!response.ok) {
    throw new Error(
      result.detail || "Clinical assessment request failed",
    );
  }

  return result;
}