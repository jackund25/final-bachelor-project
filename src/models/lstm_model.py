"""LSTM model placeholder for roadmap implementation.

This module intentionally exposes a clear not-implemented interface so the
project remains explicit about current model scope.
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional

import numpy as np

from src.models.base_model import BaseGlucoseModel


class LSTMGlucoseModel(BaseGlucoseModel):
	"""Roadmap class for future LSTM implementation."""

	def __init__(self, config: Optional[Dict] = None):
		super().__init__(config)

	def train(
		self,
		X_train: np.ndarray,
		y_train: np.ndarray,
		X_val: Optional[np.ndarray] = None,
		y_val: Optional[np.ndarray] = None,
	) -> Dict:
		raise NotImplementedError("LSTM model is planned but not implemented yet.")

	def predict(self, X: np.ndarray) -> np.ndarray:
		raise NotImplementedError("LSTM model is planned but not implemented yet.")

	def save(self, filepath: str) -> None:
		raise NotImplementedError("LSTM model is planned but not implemented yet.")

	def load(self, filepath: str) -> None:
		raise NotImplementedError("LSTM model is planned but not implemented yet.")


def main() -> None:
	parser = argparse.ArgumentParser(description="LSTM model roadmap entrypoint")
	parser.add_argument("--train", action="store_true", help="Placeholder flag for future LSTM training")
	_ = parser.parse_args()
	raise SystemExit("LSTM model is not implemented yet. Use RF baseline in src/models/rf_model.py.")


if __name__ == "__main__":
	main()
