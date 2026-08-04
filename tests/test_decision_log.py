from src.clinical_state import ClinicalDecisionLog


def test_decision_log_create_update_and_list_patients(tmp_path):
	storage_file = tmp_path / "patient_states.json"
	manager = ClinicalDecisionLog(storage_file=str(storage_file))

	state = manager.create_state("P001", {"current_glucose": 123.0, "stress_level": 7})
	updated = manager.update_state("P001", {"carbs_on_board": 20.0, "activity_level": 15})

	assert state["current_glucose"] == 123.0
	assert updated["carbs_on_board"] == 20.0
	assert updated["activity_level"] == 15
	assert manager.list_patients() == ["P001"]


def test_decision_log_persists_and_restores_state(tmp_path):
	storage_file = tmp_path / "patient_states.json"
	manager = ClinicalDecisionLog(storage_file=str(storage_file))

	manager.create_state("P002", {"current_glucose": 145.0, "stress_level": 4})
	manager.update_state("P002", {"insulin_on_board": 2.5})
	manager.save()

	reloaded = ClinicalDecisionLog(storage_file=str(storage_file))
	reloaded.load()
	state = reloaded.get_state("P002")

	assert state["current_glucose"] == 145.0
	assert state["insulin_on_board"] == 2.5
	assert state["stress_level"] == 4


def test_decision_log_records_events(tmp_path):
	storage_file = tmp_path / "patient_states.json"
	manager = ClinicalDecisionLog(storage_file=str(storage_file))

	manager.create_state("P003")
	manager.append_event("P003", "review", {"predicted_glucose": 160.0})
	manager.log_intervention("P003", "doctor_approval", "Reviewed and approved recommendation", {"risk_level": "AMAN"})

	manager.save()
	manager.load()

	assert len(manager._records["P003"].events) >= 2
	assert any(event["event_type"] == "review" for event in manager._records["P003"].events)
	assert any(event["event_type"] == "intervention" for event in manager._records["P003"].events)


def test_decision_log_persists_rag_sources_for_audit(tmp_path):
	"""Keputusan dokter harus menyimpan sumber rujukan yang ditampilkan saat itu.

	Ini yang membuat keputusan dapat ditelusuri ke halaman dokumen di kemudian hari
	(Tugas 1B butir 7), sehingga struktur bersarangnya wajib selamat proses simpan-muat.
	"""
	storage_file = tmp_path / "patient_states.json"
	manager = ClinicalDecisionLog(storage_file=str(storage_file))

	sources = [
		{
			"rank": 1,
			"kb_id": "KB-03",
			"nama_dokumen": "KB-03_PERKENI-2021_Terapi-Insulin.pdf",
			"judul_lengkap": "Pedoman Petunjuk Praktis Terapi Insulin",
			"lembaga": "PERKENI",
			"tahun": 2021,
			"halaman_pdf": 33,
			"halaman_cetak": 18,
			"page_label": "Hal. 18",
			"similarity": 0.42,
			"snippet": "Sebagai regimen awal dapat digunakan insulin basal dengan dosis 0,2 unit/kgbb",
		}
	]
	manager.log_intervention(
		"P004", "setujui rekomendasi", "Dokter menyetujui",
		{"rag_sources": sources, "rag_grounded": True},
	)
	manager.save()

	reloaded = ClinicalDecisionLog(storage_file=str(storage_file))
	reloaded.load()
	events = reloaded.get_events("P004")

	intervensi = [e for e in events if e["event_type"] == "intervention"]
	assert intervensi, "event intervensi harus tersimpan"

	tersimpan = intervensi[0]["payload"]["details"]["rag_sources"]
	assert len(tersimpan) == 1
	assert tersimpan[0]["page_label"] == "Hal. 18"
	assert tersimpan[0]["kb_id"] == "KB-03"
	assert tersimpan[0]["halaman_cetak"] == 18
