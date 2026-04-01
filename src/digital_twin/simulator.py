"""
What-If Simulation Engine
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from .patient_twin import PatientDigitalTwin

logger = logging.getLogger(__name__)


class WhatIfSimulator:
    """
    Engine for running what-if scenarios
    """
    
    def __init__(self, twin: PatientDigitalTwin):
        """
        Initialize simulator with a Digital Twin
        
        Args:
            twin: PatientDigitalTwin instance
        """
        self.twin = twin
    
    def compare_meal_scenarios(self, carb_amounts: List[float]) -> pd.DataFrame:
        """
        Compare impact of different meal sizes
        
        Args:
            carb_amounts: List of carbohydrate amounts to compare (grams)
            
        Returns:
            df_results: DataFrame with comparison results
        """
        results = []
        
        for carbs in carb_amounts:
            scenario = {
                'carbs_delta': carbs,
                'time_horizon': 120  # 2 hours
            }
            
            result = self.twin.simulate_scenario(scenario)
            
            results.append({
                'carbs_grams': carbs,
                'predicted_glucose': result['predicted_glucose'],
                'glucose_change': result['glucose_change'],
                'risk_level': result['risk_level']
            })
        
        df_results = pd.DataFrame(results)
        
        logger.info(f"Compared {len(carb_amounts)} meal scenarios")
        
        return df_results
    
    def simulate_stress_reduction(self, stress_reduction_points: int = 3) -> Dict:
        """
        Simulate impact of stress reduction (e.g., after relaxation)
        
        Args:
            stress_reduction_points: How much to reduce stress (1-10 scale)
            
        Returns:
            result: Simulation result
        """
        scenario = {
            'stress_delta': -stress_reduction_points,
            'time_horizon': 60
        }
        
        return self.twin.simulate_scenario(scenario)
    
    def simulate_exercise(self, duration_minutes: int = 30) -> Dict:
        """
        Simulate impact of physical activity
        
        Args:
            duration_minutes: Exercise duration
            
        Returns:
            result: Simulation result
        """
        scenario = {
            'activity_delta': duration_minutes,
            'time_horizon': 90
        }
        
        return self.twin.simulate_scenario(scenario)
    
    def find_optimal_insulin_dose(self, meal_carbs: float, 
                                  target_glucose: float = 120) -> Dict:
        """
        Find insulin dose to reach target glucose after meal
        
        Args:
            meal_carbs: Carbohydrate amount (grams)
            target_glucose: Desired glucose level (mg/dL)
            
        Returns:
            recommendation: Optimal dose and prediction
        """
        # Start with carb ratio estimate
        estimated_dose = meal_carbs / self.twin.parameters['carb_ratio']
        
        best_dose = estimated_dose
        best_diff = float('inf')
        
        # Try doses around estimate
        for dose in np.linspace(estimated_dose * 0.5, estimated_dose * 1.5, 20):
            scenario = {
                'carbs_delta': meal_carbs,
                'insulin_delta': dose,
                'time_horizon': 120
            }
            
            result = self.twin.simulate_scenario(scenario)
            diff = abs(result['predicted_glucose'] - target_glucose)
            
            if diff < best_diff:
                best_diff = diff
                best_dose = dose
        
        # Final prediction with best dose
        final_scenario = {
            'carbs_delta': meal_carbs,
            'insulin_delta': best_dose,
            'time_horizon': 120
        }
        
        final_result = self.twin.simulate_scenario(final_scenario)
        
        return {
            'recommended_dose': round(best_dose, 1),
            'meal_carbs': meal_carbs,
            'predicted_glucose': final_result['predicted_glucose'],
            'target_glucose': target_glucose,
            'difference': best_diff
        }
    
    def generate_decision_tree(self) -> Dict:
        """
        Generate decision support based on current state
        
        Returns:
            decision_tree: Recommended actions
        """
        current_glucose = self.twin.state['current_glucose']
        stress = self.twin.state['stress_level']
        
        decisions = {
            'current_state': self.twin.get_state_summary(),
            'recommendations': []
        }
        
        # High glucose
        if current_glucose > 180:
            decisions['recommendations'].append({
                'priority': 'HIGH',
                'action': 'Pertimbangkan insulin koreksi',
                'reason': f'Gula darah tinggi ({current_glucose:.0f} mg/dL)',
                'alternative': 'Atau lakukan aktivitas fisik ringan 20-30 menit'
            })
        
        # High stress
        if stress > 7:
            sim_result = self.simulate_stress_reduction(3)
            decisions['recommendations'].append({
                'priority': 'MEDIUM',
                'action': 'Lakukan teknik relaksasi',
                'reason': f'Stress tinggi ({stress}/10)',
                'potential_benefit': f"Dapat turunkan glukosa ~{abs(sim_result['glucose_change']):.0f} mg/dL"
            })
        
        # Low activity
        if self.twin.state['activity_level'] < 15:
            sim_result = self.simulate_exercise(30)
            decisions['recommendations'].append({
                'priority': 'MEDIUM',
                'action': 'Tingkatkan aktivitas fisik',
                'reason': 'Aktivitas hari ini kurang',
                'potential_benefit': f"30 menit jalan dapat turunkan glukosa ~{abs(sim_result['glucose_change']):.0f} mg/dL"
            })
        
        # All good
        if current_glucose >= 70 and current_glucose <= 180 and stress <= 6:
            decisions['recommendations'].append({
                'priority': 'LOW',
                'action': 'Pertahankan pola saat ini',
                'reason': 'Kondisi terkontrol dengan baik'
            })
        
        return decisions