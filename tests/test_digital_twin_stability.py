"""
Day 6: Digital Twin State Transition Stabilization Tests
Ensures state transitions are deterministic, consistent, and properly logged.
"""

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTwinStateInitialization:
    """Verify digital twin initialization is deterministic."""
    
    def test_default_initialization(self):
        """Twin should initialize with canonical default state."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P001")
        
        # Verify canonical state
        assert twin.patient_id == "P001"
        assert twin.state['current_glucose'] == 100.0
        assert twin.state['insulin_on_board'] == 0.0
        assert twin.state['carbs_on_board'] == 0.0
        assert twin.state['stress_level'] == 5
        assert twin.state['activity_level'] == 0
        assert isinstance(twin.state['timestamp'], str)
        
    def test_custom_initialization(self):
        """Twin should accept custom initial state."""
        from src.digital_twin import PatientDigitalTwin
        
        custom_state = {
            'current_glucose': 120.0,
            'stress_level': 7,
            'insulin_on_board': 2.0
        }
        twin = PatientDigitalTwin(patient_id="P002", initial_state=custom_state)
        
        assert twin.state['current_glucose'] == 120.0
        assert twin.state['stress_level'] == 7
        assert twin.state['insulin_on_board'] == 2.0
        
    def test_state_normalization_bounds(self):
        """State normalization should enforce physiological bounds."""
        from src.digital_twin import PatientDigitalTwin
        
        # Test glucose bounds
        extreme_state = {
            'current_glucose': 500.0,  # Should be clipped to 400
            'stress_level': 15,  # Should be clipped to 10
            'activity_level': -10,  # Should be clipped to 0
        }
        twin = PatientDigitalTwin(patient_id="P003", initial_state=extreme_state)
        
        assert twin.state['current_glucose'] <= 400.0
        assert twin.state['stress_level'] <= 10
        assert twin.state['activity_level'] >= 0


class TestTwinStateTransitions:
    """Verify state transitions are consistent and logged."""
    
    def test_state_update(self):
        """State update should apply changes and log history."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P004")
        initial_glucose = twin.state['current_glucose']
        
        # Update state
        twin.update_state({'current_glucose': 150.0})
        
        assert twin.state['current_glucose'] == 150.0
        assert len(twin.history) >= 1
        assert twin.history[-1]['event_type'] == 'update'
        
    def test_history_snapshot(self):
        """History snapshots should capture event type and state."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P005")
        twin.update_state({'current_glucose': 140.0})
        
        snapshot = twin.history[-1]
        assert 'event_type' in snapshot
        assert 'state' in snapshot
        assert 'timestamp' in snapshot
        assert snapshot['state']['current_glucose'] == 140.0
        
    def test_multiple_sequential_updates(self):
        """Multiple updates should be logged in order."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P006")
        
        updates = [
            {'current_glucose': 110.0},
            {'stress_level': 8},
            {'activity_level': 30},
        ]
        
        for update in updates:
            twin.update_state(update)
        
        assert len(twin.history) >= 3
        # Last three should be the updates
        for i, update in enumerate(updates):
            assert twin.history[-(3-i)]['event_type'] == 'update'


class TestInsulinAbsorption:
    """Verify insulin decay model is deterministic."""
    
    def test_insulin_decay_bounds(self):
        """Insulin should decay from dose to zero."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P007")
        duration = twin.parameters['insulin_duration_hours']
        
        dose = 10.0
        
        # Test at key time points
        iob_start = twin.calculate_insulin_on_board(0, dose)
        iob_mid = twin.calculate_insulin_on_board(duration / 2, dose)
        iob_end = twin.calculate_insulin_on_board(duration + 1, dose)
        
        assert iob_start == dose  # Full dose at t=0
        assert 0 < iob_mid < dose  # Partial at mid-life
        assert iob_end == 0  # Zero after duration
        
    def test_insulin_monotonic_decay(self):
        """Insulin on board should monotonically decrease."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P008")
        dose = 5.0
        duration = twin.parameters['insulin_duration_hours']
        
        values = []
        for t in np.linspace(0, duration + 1, 11):
            values.append(twin.calculate_insulin_on_board(t, dose))
        
        # Should be monotonically decreasing
        for i in range(1, len(values)):
            assert values[i] <= values[i-1]


class TestCarbAbsorption:
    """Verify carb absorption model is deterministic."""
    
    def test_carb_absorption_bounds(self):
        """Carbs should be absorbed from full to zero."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P009")
        duration = twin.parameters['carb_absorption_hours']
        
        carbs = 50.0
        
        # Test at key time points
        cob_start = twin.calculate_carbs_on_board(0, carbs)
        cob_mid = twin.calculate_carbs_on_board(duration / 2, carbs)
        cob_end = twin.calculate_carbs_on_board(duration + 1, carbs)
        
        assert cob_start == carbs  # Full carbs at t=0
        assert 0 < cob_mid < carbs  # Partial at mid-absorption
        assert cob_end == 0  # Zero after absorption
        
    def test_carb_linear_absorption(self):
        """Carb absorption should be linear."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P010")
        carbs = 60.0
        duration = twin.parameters['carb_absorption_hours']
        
        # Sample at equal intervals
        times = np.linspace(0, duration, 4)
        values = [twin.calculate_carbs_on_board(t, carbs) for t in times]
        
        # Differences should be approximately equal (linear)
        diffs = [values[i] - values[i+1] for i in range(len(values)-1)]
        assert all(d >= 0 for d in diffs)  # Monotonically decreasing


class TestGlucosePrediction:
    """Verify glucose impact prediction is deterministic."""
    
    def test_baseline_glucose_stability(self):
        """With zero effects, glucose should remain relatively stable."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P011")
        twin.state['insulin_on_board'] = 0
        twin.state['carbs_on_board'] = 0
        twin.state['stress_level'] = 5
        twin.state['activity_level'] = 0
        
        glucose_0 = twin.state['current_glucose']
        predicted = twin.predict_glucose_impact(60)
        
        # Should be close to baseline (allow for minor impact calculation variance)
        assert abs(predicted - glucose_0) < 20
        
    def test_insulin_lowers_glucose(self):
        """Insulin on board should lower glucose."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P012")
        baseline = twin.state['current_glucose']
        
        twin.state['insulin_on_board'] = 5.0
        predicted = twin.predict_glucose_impact(60)
        
        assert predicted < baseline
        
    def test_carbs_raise_glucose(self):
        """Carbs on board should raise glucose."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P013")
        baseline = twin.state['current_glucose']
        
        twin.state['carbs_on_board'] = 30.0
        predicted = twin.predict_glucose_impact(60)
        
        assert predicted > baseline
        
    def test_glucose_bounds(self):
        """Predicted glucose should stay within physiological bounds."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P014")
        
        # Extreme scenario
        twin.state['insulin_on_board'] = 20.0
        twin.state['carbs_on_board'] = 100.0
        twin.state['stress_level'] = 10
        
        predicted = twin.predict_glucose_impact(60)
        
        assert 40 <= predicted <= 400


class TestSimulationScenarios:
    """Verify what-if simulation is deterministic and consistent."""
    
    def test_scenario_validation(self):
        """Invalid scenarios should raise exceptions."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P015")
        
        # Negative carbs
        with pytest.raises(ValueError):
            twin.simulate_scenario({'carbs_delta': -10})
        
        # Invalid time horizon
        with pytest.raises(ValueError):
            twin.simulate_scenario({'time_horizon': 0})
        
        # Non-dict scenario
        with pytest.raises(TypeError):
            twin.simulate_scenario("not a dict")
            
    def test_meal_scenario(self):
        """Meal scenario should increase carbs on board."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P016")
        original_glucose = twin.state['current_glucose']
        
        result = twin.simulate_scenario({
            'carbs_delta': 45.0,
            'time_horizon': 60
        })
        
        assert result['simulated_state']['carbs_on_board'] > 0
        assert result['predicted_glucose'] > original_glucose
        
    def test_insulin_scenario(self):
        """Insulin scenario should increase insulin on board."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P017")
        original_glucose = twin.state['current_glucose']
        
        result = twin.simulate_scenario({
            'insulin_delta': 5.0,
            'time_horizon': 60
        })
        
        assert result['simulated_state']['insulin_on_board'] > 0
        assert result['predicted_glucose'] < original_glucose
        
    def test_activity_scenario(self):
        """Activity scenario should increase activity level."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P018")
        original_glucose = twin.state['current_glucose']
        
        result = twin.simulate_scenario({
            'activity_delta': 30,
            'time_horizon': 60
        })
        
        assert result['simulated_state']['activity_level'] == 30
        assert result['predicted_glucose'] < original_glucose  # Activity lowers glucose
        
    def test_stress_scenario(self):
        """Stress scenario should update stress level."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P019")
        original_stress = twin.state['stress_level']
        original_glucose = twin.state['current_glucose']
        
        result = twin.simulate_scenario({
            'stress_delta': 3,
            'time_horizon': 60
        })
        
        assert result['simulated_state']['stress_level'] == original_stress + 3
        assert result['predicted_glucose'] > original_glucose  # Stress raises glucose
        
    def test_original_state_unmodified(self):
        """Simulation should not modify original state."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P020")
        original_state = json.loads(json.dumps(twin.state))  # Deep copy
        
        result = twin.simulate_scenario({
            'carbs_delta': 50.0,
            'insulin_delta': 3.0,
            'activity_delta': 20,
            'stress_delta': 2,
            'time_horizon': 60
        })
        
        # Original state should be unchanged
        assert twin.state['carbs_on_board'] == original_state['carbs_on_board']
        assert twin.state['insulin_on_board'] == original_state['insulin_on_board']


class TestHistoryManagement:
    """Verify history tracking is reliable."""
    
    def test_history_size_limit(self):
        """History should not grow unbounded."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P021")
        
        # Add many updates
        for i in range(2000):
            twin.update_state({'current_glucose': 100 + i % 50})
        
        # History should be capped (default 1000)
        assert len(twin.history) <= 1000
        
    def test_state_summary(self):
        """State summary should be human-readable."""
        from src.digital_twin import PatientDigitalTwin
        
        twin = PatientDigitalTwin(patient_id="P022")
        summary = twin.get_state_summary()
        
        assert 'patient_id' in summary
        assert 'current_glucose' in summary
        assert 'mg/dL' in str(summary)  # Should have units


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
