from src.digital_twin import DigitalTwinStateManager


def test_state_manager_create_update_and_list_patients(tmp_path):
	storage_file = tmp_path / "patient_states.json"
	manager = DigitalTwinStateManager(storage_file=str(storage_file))

	state = manager.create_state("P001", {"current_glucose": 123.0, "stress_level": 7})
	updated = manager.update_state("P001", {"carbs_on_board": 20.0, "activity_level": 15})

	assert state["current_glucose"] == 123.0
	assert updated["carbs_on_board"] == 20.0
	assert updated["activity_level"] == 15
	assert manager.list_patients() == ["P001"]


def test_state_manager_persists_and_restores_state(tmp_path):
	storage_file = tmp_path / "patient_states.json"
	manager = DigitalTwinStateManager(storage_file=str(storage_file))

	manager.create_state("P002", {"current_glucose": 145.0, "stress_level": 4})
	manager.update_state("P002", {"insulin_on_board": 2.5})
	manager.save()

	reloaded = DigitalTwinStateManager(storage_file=str(storage_file))
	reloaded.load()
	state = reloaded.get_state("P002")

	assert state["current_glucose"] == 145.0
	assert state["insulin_on_board"] == 2.5
	assert state["stress_level"] == 4


def test_state_manager_records_events(tmp_path):
	storage_file = tmp_path / "patient_states.json"
	manager = DigitalTwinStateManager(storage_file=str(storage_file))

	manager.create_state("P003")
	manager.append_event("P003", "simulation", {"predicted_glucose": 160.0})

	manager.save()
	manager.load()

	assert len(manager._records["P003"].events) >= 2
	assert manager._records["P003"].events[-1]["event_type"] == "simulation"