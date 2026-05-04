"""State management for the patient digital twin lifecycle."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import os
from typing import Any, Dict, Optional


DEFAULT_TWIN_STATE: Dict[str, Any] = {
	"current_glucose": 100.0,
	"insulin_on_board": 0.0,
	"carbs_on_board": 0.0,
	"last_meal_time": None,
	"last_insulin_time": None,
	"activity_level": 0,
	"stress_level": 5,
	"timestamp": None,
}


def _now_iso() -> str:
	return datetime.now().isoformat()


def _normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
	normalized = deepcopy(DEFAULT_TWIN_STATE)
	normalized.update(state)
	if not normalized.get("timestamp"):
		normalized["timestamp"] = _now_iso()
	return normalized


def _to_jsonable(value: Any) -> Any:
	"""Convert runtime objects to JSON-safe primitives recursively."""
	if value is None or isinstance(value, (str, int, float, bool)):
		return value

	if isinstance(value, datetime):
		return value.isoformat()

	if isinstance(value, dict):
		return {key: _to_jsonable(item) for key, item in value.items()}

	if isinstance(value, (list, tuple, set)):
		return [_to_jsonable(item) for item in value]

	# Covers pandas.Timestamp and similar datetime-like objects.
	if hasattr(value, "isoformat"):
		try:
			return value.isoformat()
		except Exception:
			pass

	# Covers numpy scalar objects.
	if hasattr(value, "item"):
		try:
			return value.item()
		except Exception:
			pass

	return str(value)


@dataclass
class StateRecord:
	"""Persisted state bundle for a single patient."""

	patient_id: str
	state: Dict[str, Any] = field(default_factory=dict)
	events: list[Dict[str, Any]] = field(default_factory=list)


class DigitalTwinStateManager:
	"""Manage digital twin state for one or more patients."""

	def __init__(self, storage_file: str = "data/processed/patient_states.json"):
		self.storage_path = Path(storage_file)
		self.storage_path.parent.mkdir(parents=True, exist_ok=True)
		self._records: Dict[str, StateRecord] = {}

	def create_state(self, patient_id: str, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		"""Create a new twin state or return the existing one."""
		if patient_id in self._records:
			return deepcopy(self._records[patient_id].state)

		state = _normalize_state(initial_state or {})
		self._records[patient_id] = StateRecord(patient_id=patient_id, state=state, events=[])
		self._append_event(patient_id, "create", state)
		return deepcopy(state)

	def get_state(self, patient_id: str) -> Dict[str, Any]:
		"""Return a copy of the current state for a patient."""
		if patient_id not in self._records:
			raise KeyError(f"Patient state not found: {patient_id}")
		return deepcopy(self._records[patient_id].state)

	def update_state(self, patient_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
		"""Merge updates into the patient state and record an event."""
		if patient_id not in self._records:
			self.create_state(patient_id)

		record = self._records[patient_id]
		previous_state = deepcopy(record.state)
		for key, value in updates.items():
			if key in DEFAULT_TWIN_STATE:
				record.state[key] = value

		record.state["timestamp"] = _now_iso()
		self._append_event(patient_id, "update", updates, previous_state=previous_state)
		return deepcopy(record.state)

	def append_event(self, patient_id: str, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
		"""Append an arbitrary lifecycle event for a patient."""
		if patient_id not in self._records:
			self.create_state(patient_id)
		return self._append_event(patient_id, event_type, payload)

	def list_patients(self) -> list[str]:
		"""List all known patient IDs."""
		return sorted(self._records.keys())

	def save(self) -> None:
		"""Persist all states and events to disk."""
		payload = {
			patient_id: {
				"patient_id": record.patient_id,
				"state": _to_jsonable(record.state),
				"events": _to_jsonable(record.events),
			}
			for patient_id, record in self._records.items()
		}
		# Write atomically: write to a temp file then replace the target
		tmp_path = self.storage_path.with_name(self.storage_path.name + ".tmp")
		with open(tmp_path, "w", encoding="utf-8") as file_handle:
			json.dump(payload, file_handle, ensure_ascii=False, indent=2)
		# Use os.replace for an atomic rename on most platforms
		os.replace(tmp_path, self.storage_path)

	def load(self) -> None:
		"""Load persisted states from disk if present."""
		if not self.storage_path.exists():
			return

		try:
			with open(self.storage_path, "r", encoding="utf-8") as file_handle:
				payload = json.load(file_handle) or {}
		except json.JSONDecodeError as exc:
			# Backup the corrupted file so the user can inspect/restore it
			backup_path = self.storage_path.with_name(self.storage_path.name + f".corrupt-{_now_iso()}")
			try:
				self.storage_path.replace(backup_path)
			except Exception:
				# fallback to copy if replace fails
				try:
					with open(backup_path, "w", encoding="utf-8") as out_f, open(self.storage_path, "r", encoding="utf-8", errors="ignore") as in_f:
						out_f.write(in_f.read())
				except Exception:
					pass
			print(f"Warning: corrupted state file moved to {backup_path}; starting with empty state. Error: {exc}")
			payload = {}

		self._records = {}
		for patient_id, record in payload.items():
			state = _normalize_state(record.get("state", {}))
			events = record.get("events", []) or []
			self._records[patient_id] = StateRecord(patient_id=patient_id, state=state, events=events)

	def _append_event(
		self,
		patient_id: str,
		event_type: str,
		payload: Dict[str, Any],
		previous_state: Optional[Dict[str, Any]] = None,
	) -> Dict[str, Any]:
		record = self._records[patient_id]
		event = {
			"event_type": event_type,
			"payload": deepcopy(payload),
			"timestamp": _now_iso(),
		}
		if previous_state is not None:
			event["previous_state"] = previous_state
		record.events.append(event)
		return event
