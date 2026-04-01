"""
Data Generator using Simglucose
Generates synthetic patient data with glucose, carbs, insulin, activity, and stress
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import logging
from tqdm import tqdm

try:
    # Optional dependency for real physiological simulation
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.actuator.pump import InsulinPump
    from simglucose.simulation.scenario_gen import RandomScenario
    from simglucose.controller.base import Action
    SIMGLUCOSE_AVAILABLE = True
except ImportError:
    T1DSimEnv = None
    T1DPatient = None
    CGMSensor = None
    InsulinPump = None
    RandomScenario = None
    Action = None
    SIMGLUCOSE_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiabetesDataGenerator:
    """
    Generate synthetic diabetes patient data
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize generator with config
        
        Args:
            config_path: Path to YAML config file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.patient_config = self.config['patient']
        
        # Create output directory
        self.output_dir = Path(self.data_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default generation mode uses dummy data so development can proceed
        # even when Simglucose or its dependencies are unavailable.
        self.default_mode = self.data_config.get('generation_mode', 'dummy')
        
        logger.info("DataGenerator initialized")
        logger.info(f"Config: {self.data_config['num_patients']} patients, "
                   f"{self.data_config['days_per_patient']} days each")
    
    def _generate_stress_pattern(self, hour):
        """
        Generate realistic stress level based on time of day
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            stress_level: 1-10 scale
        """
        if 6 <= hour < 9:  # Morning rush
            base = 7
            variance = 1
        elif 12 <= hour < 14:  # Lunch break
            base = 4
            variance = 1
        elif 17 <= hour < 19:  # Evening commute
            base = 7
            variance = 1
        elif 22 <= hour or hour < 6:  # Sleep
            base = 2
            variance = 1
        else:  # Regular work hours
            base = 5
            variance = 2
        
        stress = np.random.randint(
            max(1, base - variance), 
            min(10, base + variance + 1)
        )
        return stress
    
    def _generate_activity_pattern(self, hour, stress):
        """
        Generate activity level based on time and stress
        
        Args:
            hour: Hour of day
            stress: Current stress level
            
        Returns:
            activity_minutes: Physical activity in minutes
        """
        # Less activity during high stress
        if stress > 7:
            prob_active = 0.1
        elif 22 <= hour or hour < 6:  # Sleep hours
            return 0
        elif 6 <= hour < 9 or 17 <= hour < 19:  # Commute
            prob_active = 0.2
        else:
            prob_active = 0.3
        
        if np.random.random() < prob_active:
            # Activity duration
            return np.random.choice([15, 30, 45, 60], p=[0.5, 0.3, 0.15, 0.05])
        else:
            return 0
    
    def _generate_meal_pattern(self, hour):
        """
        Generate meal carbohydrate intake based on meal times
        
        Args:
            hour: Hour of day
            
        Returns:
            carbs_grams: Carbohydrate intake in grams
        """
        # Meal times
        if 7 <= hour < 9:  # Breakfast
            if np.random.random() < 0.8:  # 80% chance
                return np.random.choice([30, 45, 60], p=[0.3, 0.5, 0.2])
        elif 12 <= hour < 14:  # Lunch
            if np.random.random() < 0.9:
                return np.random.choice([45, 60, 75], p=[0.3, 0.5, 0.2])
        elif 18 <= hour < 20:  # Dinner
            if np.random.random() < 0.85:
                return np.random.choice([45, 60, 75, 90], p=[0.2, 0.4, 0.3, 0.1])
        
        # Snacks (random throughout day)
        if 9 <= hour < 22:
            if np.random.random() < 0.05:  # 5% chance each 5-min
                return np.random.choice([15, 20, 30], p=[0.5, 0.3, 0.2])
        
        return 0

    def _generate_sleep_flag(self, hour: int) -> int:
        """Return 1 during sleeping hours, otherwise 0."""
        return int(hour >= 22 or hour < 6)

    def _generate_work_flag(self, hour: int) -> int:
        """Return 1 during typical working hours, otherwise 0."""
        return int(9 <= hour < 17)

    def _generate_illness_flag(self, rng: np.random.Generator) -> int:
        """Occasional illness flag to create realistic perturbations."""
        return int(rng.random() < 0.03)

    def _simulate_glucose_step(
        self,
        current_glucose: float,
        carbs: float,
        insulin: float,
        activity: int,
        stress: int,
        sleep_flag: int,
        work_flag: int,
        illness_flag: int,
        rng: np.random.Generator,
    ) -> float:
        """Generate a realistic next glucose value for dummy data."""
        baseline = float(self.patient_config.get('baseline_glucose', 100))
        glucose = current_glucose

        # Gentle mean reversion keeps the series physiologically plausible.
        glucose += (baseline - current_glucose) * 0.15
        glucose += carbs * 1.2
        glucose -= insulin * self.patient_config['insulin_sensitivity'] * 0.08
        glucose -= activity * 0.05
        glucose += (stress - 5) * 0.6
        glucose -= sleep_flag * 0.3
        glucose += work_flag * 0.5
        glucose += illness_flag * 3.0
        glucose += rng.normal(0, 2.5)
        return float(np.clip(glucose, 40, 400))

    def _generate_dummy_patient_data(self, patient_name, days=7, seed=None):
        """Generate realistic synthetic data without using Simglucose."""
        rng = np.random.default_rng(seed)
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        intervals = days * 24 * 60 // self.data_config.get('sampling_interval_min', 5)
        time_step = timedelta(minutes=self.data_config.get('sampling_interval_min', 5))

        records = []
        current_glucose = float(self.patient_config.get('baseline_glucose', 100))
        insulin_on_board = 0.0
        carbs_on_board = 0.0
        last_glucose = current_glucose

        for step in range(intervals):
            current_time = start_time + step * time_step
            hour = current_time.hour

            stress = int(self._generate_stress_pattern(hour))
            activity = int(self._generate_activity_pattern(hour, stress))
            carbs = float(self._generate_meal_pattern(hour))
            sleep_flag = self._generate_sleep_flag(hour)
            work_flag = self._generate_work_flag(hour)
            illness_flag = self._generate_illness_flag(rng)

            if carbs > 0:
                insulin = float(round(carbs / self.patient_config['carb_ratio'], 2))
            else:
                insulin = float(round(max(0.0, rng.normal(0.05, 0.08)), 2)) if work_flag else 0.0

            carbs_on_board = max(0.0, carbs_on_board * 0.72 + carbs)
            insulin_on_board = max(0.0, insulin_on_board * 0.78 + insulin)

            current_glucose = self._simulate_glucose_step(
                current_glucose=last_glucose,
                carbs=carbs_on_board,
                insulin=insulin_on_board,
                activity=activity,
                stress=stress,
                sleep_flag=sleep_flag,
                work_flag=work_flag,
                illness_flag=illness_flag,
                rng=rng,
            )
            last_glucose = current_glucose

            meal_type = 'snack'
            if carbs >= 70:
                meal_type = 'dinner'
            elif carbs >= 50:
                meal_type = 'lunch'
            elif carbs >= 25:
                meal_type = 'breakfast'

            record = {
                'patient_id': patient_name,
                'timestamp': current_time,
                'glucose': round(current_glucose, 1),
                'carbs': round(carbs, 1),
                'insulin': round(insulin, 2),
                'activity': activity,
                'stress': stress,
                'sleep': sleep_flag,
                'work': work_flag,
                'illness': illness_flag,
                'meal_type': meal_type,
            }
            records.append(record)

        df = pd.DataFrame(records)
        df['glucose_change'] = df.groupby('patient_id')['glucose'].diff().fillna(0.0)
        return df
    
    def generate_patient_data(self, patient_name, days=7, seed=None):
        """
        Generate data for single patient
        
        Args:
            patient_name: Patient ID (e.g., 'adult#001')
            days: Number of days to simulate
            seed: Random seed for reproducibility
            
        Returns:
            df: DataFrame with patient data
        """
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(f"Generating data for {patient_name}...")

        use_dummy = self.default_mode == 'dummy' or not SIMGLUCOSE_AVAILABLE

        if use_dummy:
            df = self._generate_dummy_patient_data(patient_name, days=days, seed=seed)
            logger.info(f"✓ Generated {len(df)} dummy records for {patient_name}")
            return df
        
        # Initialize Simglucose environment
        patient = T1DPatient.withName(patient_name)
        sensor = CGMSensor.withName('Dexcom', seed=seed)
        pump = InsulinPump.withName('Insulet')
        scenario = RandomScenario(
            start_time=datetime(2024, 1, 1, 0, 0, 0),
            seed=seed
        )
        
        env = T1DSimEnv(patient, sensor, pump, scenario)
        obs = env.reset()
        
        # Calculate number of 5-minute intervals
        intervals = days * 24 * 60 // 5
        
        records = []
        
        for step in tqdm(range(intervals), desc=f"Patient {patient_name}"):
            current_time = env.time
            hour = current_time.hour
            
            # Generate contextual variables
            stress = self._generate_stress_pattern(hour)
            activity = self._generate_activity_pattern(hour, stress)
            carbs = self._generate_meal_pattern(hour)
            
            # Calculate insulin dose (simplified bolus calculation)
            if carbs > 0:
                insulin = carbs / self.patient_config['carb_ratio']
            else:
                insulin = 0
            
            # Step simulation
            action = Action(basal=0, bolus=insulin)
            obs, reward, done, info = env.step(action)
            
            # Record data
            record = {
                'patient_id': patient_name,
                'timestamp': current_time,
                'glucose': obs.CGM,  # mg/dL
                'carbs': carbs,
                'insulin': insulin,
                'activity': activity,
                'stress': stress,
                'sleep': self._generate_sleep_flag(hour),
                'work': self._generate_work_flag(hour),
                'illness': self._generate_illness_flag(np.random.default_rng(seed)),
                'meal_type': 'meal' if carbs > 0 else 'none'
            }
            records.append(record)
            
            if done:
                logger.warning(f"Simulation ended early at step {step}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        logger.info(f"✓ Generated {len(df)} records for {patient_name}")
        logger.info(f"  Glucose range: {df['glucose'].min():.1f} - {df['glucose'].max():.1f} mg/dL")
        logger.info(f"  Mean glucose: {df['glucose'].mean():.1f} mg/dL")
        
        return df
    
    def generate_dataset(self, save=True):
        """
        Generate complete dataset for all patients
        
        Args:
            save: Whether to save to CSV
            
        Returns:
            df_combined: Combined DataFrame for all patients
        """
        num_patients = self.data_config['num_patients']
        days = self.data_config['days_per_patient']
        seed = self.data_config['seed']
        
        logger.info("="*60)
        logger.info("STARTING DATASET GENERATION")
        logger.info("="*60)
        
        all_data = []
        
        # Get available patient names from Simglucose
        patient_names = [f'adult#{str(i).zfill(3)}' for i in range(1, num_patients + 1)]
        
        for i, patient_name in enumerate(patient_names):
            # Use different seed for each patient
            patient_seed = seed + i if seed else None
            
            df_patient = self.generate_patient_data(
                patient_name=patient_name,
                days=days,
                seed=patient_seed
            )
            
            all_data.append(df_patient)
        
        # Combine all patients
        df_combined = pd.concat(all_data, ignore_index=True)
        
        logger.info("="*60)
        logger.info("GENERATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total records: {len(df_combined):,}")
        logger.info(f"Date range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")
        logger.info(f"Memory usage: {df_combined.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        
        # Save
        if save:
            output_file = self.output_dir / 'training_data_complete.csv'
            df_combined.to_csv(output_file, index=False)
            logger.info(f"✓ Saved to {output_file}")
            
            # Also save metadata
            metadata = {
                'generation_date': datetime.now().isoformat(),
                'num_patients': num_patients,
                'days_per_patient': days,
                'total_records': len(df_combined),
                'features': list(df_combined.columns),
                'glucose_stats': {
                    'mean': float(df_combined['glucose'].mean()),
                    'std': float(df_combined['glucose'].std()),
                    'min': float(df_combined['glucose'].min()),
                    'max': float(df_combined['glucose'].max())
                }
            }
            
            import json
            metadata_file = self.output_dir / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✓ Saved metadata to {metadata_file}")
        
        return df_combined


def main():
    """Main execution"""
    generator = DiabetesDataGenerator('config.yaml')
    df = generator.generate_dataset(save=True)
    
    print("\n" + "="*60)
    print("QUICK PREVIEW")
    print("="*60)
    print(df.head(10))
    print("\n" + df.describe(include='all').transpose().to_string())


if __name__ == "__main__":
    main()