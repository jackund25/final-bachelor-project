"""
Patient Digital Twin - Virtual representation of patient's metabolic state
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PatientDigitalTwin:
    """
    Digital Twin representation of a diabetes patient
    
    Maintains current state and enables what-if simulation
    """
    
    def __init__(self, patient_id: str, initial_state: Optional[Dict] = None):
        """
        Initialize Digital Twin
        
        Args:
            patient_id: Unique patient identifier
            initial_state: Initial metabolic state (optional)
        """
        self.patient_id = patient_id
        
        # Default initial state
        default_state = {
            'current_glucose': 100.0,
            'insulin_on_board': 0.0,
            'carbs_on_board': 0.0,
            'last_meal_time': None,
            'last_insulin_time': None,
            'activity_level': 0,
            'stress_level': 5,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update with provided initial state
        if initial_state:
            default_state.update(initial_state)
        
        self.state = default_state
        self.history = []
        
        # Patient-specific parameters (could be learned from data)
        self.parameters = {
            'insulin_sensitivity': 50.0,  # mg/dL per unit
            'carb_ratio': 10.0,  # grams per unit
            'insulin_duration_hours': 4.0,
            'carb_absorption_hours': 3.0,
            'stress_glucose_impact': 2.0,  # mg/dL per stress point
            'activity_glucose_impact': -1.5,  # mg/dL per minute
        }
        
        logger.info(f"Digital Twin initialized for patient {patient_id}")
    
    def update_state(self, new_data: Dict) -> None:
        """
        Update twin state with new measurements or predictions
        
        Args:
            new_data: Dictionary with updated values
        """
        # Save current state to history
        self.history.append(self.state.copy())
        
        # Limit history size to prevent memory issues
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        # Update state
        for key, value in new_data.items():
            if key in self.state:
                self.state[key] = value
        
        self.state['timestamp'] = datetime.now().isoformat()
        
        logger.debug(f"State updated for {self.patient_id}: {new_data}")
    
    def calculate_insulin_on_board(self, hours_since_dose: float, dose: float) -> float:
        """
        Calculate active insulin using exponential decay model
        
        Args:
            hours_since_dose: Hours since insulin was administered
            dose: Initial insulin dose (units)
            
        Returns:
            active_insulin: Remaining active insulin (units)
        """
        duration = self.parameters['insulin_duration_hours']
        
        if hours_since_dose >= duration:
            return 0.0
        
        # Exponential decay
        decay_rate = np.log(2) / (duration / 2)  # Half-life based
        active_insulin = dose * np.exp(-decay_rate * hours_since_dose)
        
        return active_insulin
    
    def calculate_carbs_on_board(self, hours_since_meal: float, carbs: float) -> float:
        """
        Calculate unabsorbed carbohydrates
        
        Args:
            hours_since_meal: Hours since carbs were consumed
            carbs: Initial carbohydrate amount (grams)
            
        Returns:
            active_carbs: Remaining unabsorbed carbs (grams)
        """
        duration = self.parameters['carb_absorption_hours']
        
        if hours_since_meal >= duration:
            return 0.0
        
        # Linear absorption model (simplified)
        absorption_rate = carbs / duration
        absorbed = absorption_rate * hours_since_meal
        active_carbs = max(0, carbs - absorbed)
        
        return active_carbs
    
    def predict_glucose_impact(self, time_horizon_minutes: int = 60) -> float:
        """
        Predict glucose change based on current state
        
        Args:
            time_horizon_minutes: Prediction horizon in minutes
            
        Returns:
            predicted_glucose: Estimated glucose level
        """
        current_glucose = self.state['current_glucose']
        
        # Calculate impacts
        insulin_impact = -self.state['insulin_on_board'] * self.parameters['insulin_sensitivity']
        carb_impact = self.state['carbs_on_board'] * 3  # Simplified: 1g carb = 3 mg/dL
        stress_impact = self.state['stress_level'] * self.parameters['stress_glucose_impact']
        activity_impact = self.state['activity_level'] * self.parameters['activity_glucose_impact']
        
        # Total predicted change
        total_impact = insulin_impact + carb_impact + stress_impact + activity_impact
        
        # Scale by time horizon (linear approximation)
        time_factor = time_horizon_minutes / 60.0
        scaled_impact = total_impact * time_factor
        
        predicted_glucose = current_glucose + scaled_impact
        
        # Physiological bounds
        predicted_glucose = np.clip(predicted_glucose, 40, 400)
        
        return predicted_glucose
    
    def simulate_scenario(self, scenario: Dict) -> Dict:
        """
        What-if analysis: simulate impact of hypothetical changes
        
        Args:
            scenario: Dictionary with changes to simulate
                - 'carbs_delta': Additional carbs (grams)
                - 'insulin_delta': Additional insulin (units)
                - 'stress_delta': Change in stress level
                - 'activity_delta': Additional activity (minutes)
                - 'time_horizon': Prediction window (minutes)
                
        Returns:
            simulation_result: Predicted state after changes
        """
        logger.info(f"Simulating scenario for {self.patient_id}: {scenario}")
        
        # Clone current state
        simulated_state = self.state.copy()
        
        # Apply scenario changes
        if 'carbs_delta' in scenario:
            simulated_state['carbs_on_board'] += scenario['carbs_delta']
        
        if 'insulin_delta' in scenario:
            simulated_state['insulin_on_board'] += scenario['insulin_delta']
        
        if 'stress_delta' in scenario:
            new_stress = simulated_state['stress_level'] + scenario['stress_delta']
            simulated_state['stress_level'] = np.clip(new_stress, 1, 10)
        
        if 'activity_delta' in scenario:
            simulated_state['activity_level'] += scenario['activity_delta']
        
        # Get time horizon
        time_horizon = scenario.get('time_horizon', 60)
        
        # Calculate predicted glucose
        # (Temporarily update state for prediction)
        original_state = self.state.copy()
        self.state = simulated_state
        
        predicted_glucose = self.predict_glucose_impact(time_horizon)
        simulated_state['predicted_glucose'] = predicted_glucose
        
        # Restore original state
        self.state = original_state
        
        # Calculate risk level
        if predicted_glucose < 70:
            risk_level = "BAHAYA - Hipoglikemia"
            risk_color = "red"
        elif predicted_glucose > 180:
            risk_level = "HATI-HATI - Hiperglikemia"
            risk_color = "orange"
        else:
            risk_level = "AMAN"
            risk_color = "green"
        
        result = {
            'scenario': scenario,
            'current_glucose': self.state['current_glucose'],
            'predicted_glucose': predicted_glucose,
            'glucose_change': predicted_glucose - self.state['current_glucose'],
            'risk_level': risk_level,
            'risk_color': risk_color,
            'simulated_state': simulated_state,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Simulation result: {predicted_glucose:.1f} mg/dL ({risk_level})")
        
        return result
    
    def get_state_summary(self) -> Dict:
        """
        Get human-readable state summary
        
        Returns:
            summary: Dictionary with formatted state info
        """
        return {
            'patient_id': self.patient_id,
            'current_glucose': f"{self.state['current_glucose']:.1f} mg/dL",
            'insulin_active': f"{self.state['insulin_on_board']:.2f} units",
            'carbs_active': f"{self.state['carbs_on_board']:.1f} grams",
            'stress_level': f"{self.state['stress_level']}/10",
            'activity_level': f"{self.state['activity_level']} minutes",
            'last_updated': self.state['timestamp']
        }
    
    def export_state(self, filepath: Optional[Path] = None) -> str:
        """
        Export current state to JSON
        
        Args:
            filepath: Optional file path to save (if None, returns JSON string)
            
        Returns:
            json_str: JSON representation of state
        """
        export_data = {
            'patient_id': self.patient_id,
            'state': self.state,
            'parameters': self.parameters,
            'history_length': len(self.history)
        }
        
        json_str = json.dumps(export_data, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"State exported to {filepath}")
        
        return json_str
    
    @classmethod
    def load_from_json(cls, filepath: Path) -> 'PatientDigitalTwin':
        """
        Load Digital Twin from saved JSON
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            twin: Restored PatientDigitalTwin instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        twin = cls(patient_id=data['patient_id'], initial_state=data['state'])
        twin.parameters = data['parameters']
        
        logger.info(f"Digital Twin loaded from {filepath}")
        
        return twin


class TwinManager:
    """
    Manage multiple patient Digital Twins
    """
    
    def __init__(self, storage_dir: str = "data/processed/twins"):
        """
        Initialize Twin Manager
        
        Args:
            storage_dir: Directory to store twin states
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.twins: Dict[str, PatientDigitalTwin] = {}
        
        logger.info(f"TwinManager initialized with storage at {storage_dir}")
    
    def create_twin(self, patient_id: str, initial_state: Optional[Dict] = None
                   ) -> PatientDigitalTwin:
        """
        Create new Digital Twin
        
        Args:
            patient_id: Patient ID
            initial_state: Initial state
            
        Returns:
            twin: New PatientDigitalTwin instance
        """
        if patient_id in self.twins:
            logger.warning(f"Twin for {patient_id} already exists, returning existing")
            return self.twins[patient_id]
        
        twin = PatientDigitalTwin(patient_id, initial_state)
        self.twins[patient_id] = twin
        
        return twin
    
    def get_twin(self, patient_id: str) -> Optional[PatientDigitalTwin]:
        """
        Retrieve existing twin
        
        Args:
            patient_id: Patient ID
            
        Returns:
            twin: PatientDigitalTwin or None if not found
        """
        return self.twins.get(patient_id)
    
    def save_all_twins(self) -> None:
        """
        Save all twins to disk
        """
        for patient_id, twin in self.twins.items():
            filepath = self.storage_dir / f"{patient_id}.json"
            twin.export_state(filepath)
        
        logger.info(f"Saved {len(self.twins)} twins to {self.storage_dir}")
    
    def load_all_twins(self) -> None:
        """
        Load all twins from disk
        """
        json_files = list(self.storage_dir.glob("*.json"))
        
        for filepath in json_files:
            twin = PatientDigitalTwin.load_from_json(filepath)
            self.twins[twin.patient_id] = twin
        
        logger.info(f"Loaded {len(self.twins)} twins from {self.storage_dir}")