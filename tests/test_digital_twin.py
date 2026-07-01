from src.digital_twin import PatientDigitalTwin


def test_patient_digital_twin_initialization_defaults():
	twin = PatientDigitalTwin(patient_id="P001")

	assert twin.patient_id == "P001"
	assert twin.state["current_glucose"] == 100.0
	assert "timestamp" in twin.state


def test_simulate_scenario_contains_required_keys():
	twin = PatientDigitalTwin(
		patient_id="P002",
		initial_state={"current_glucose": 120.0, "stress_level": 5},
	)

	result = twin.simulate_scenario({"carbs_delta": 20, "time_horizon": 60})

	assert "predicted_glucose" in result
	assert "risk_level" in result
	assert "glucose_change" in result
	assert twin.history[-1]["event_type"] == "simulation"


def test_simulate_scenario_rejects_invalid_inputs():
	twin = PatientDigitalTwin(patient_id="P003")

	import pytest

	with pytest.raises(ValueError, match="time_horizon must be greater than 0"):
		twin.simulate_scenario({"time_horizon": 0})

	with pytest.raises(ValueError, match="carbs_delta cannot be negative"):
		twin.simulate_scenario({"carbs_delta": -1, "time_horizon": 60})
