#!/usr/bin/env python3
"""
Day 5: Verify and lock prediction artifacts (models + metrics).
Ensures reproducibility and provides artifact validation checklist.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def verify_model_artifacts():
    """Verify all expected model pickle files exist and are loadable."""
    required_models = ["rf_baseline.pkl", "gb_baseline.pkl"]
    logger.info("=" * 70)
    logger.info("MODEL ARTIFACT VERIFICATION")
    logger.info("=" * 70)
    
    all_ok = True
    for model_name in required_models:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            logger.error(f"❌ {model_name}: NOT FOUND")
            all_ok = False
            continue
        
        try:
            model = joblib.load(model_path)
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ {model_name}: LOADED ({file_size_mb:.2f} MB)")
        except Exception as e:
            logger.error(f"❌ {model_name}: FAILED TO LOAD ({str(e)})")
            all_ok = False
    
    return all_ok


def verify_metrics_file():
    """Verify metrics_experiments.json exists and is valid JSON."""
    metrics_path = MODELS_DIR / "metrics_experiments.json"
    logger.info("=" * 70)
    logger.info("METRICS FILE VERIFICATION")
    logger.info("=" * 70)
    
    if not metrics_path.exists():
        logger.error("❌ metrics_experiments.json: NOT FOUND")
        return False, None
    
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        # Validate structure
        required_keys = ["results", "config"]
        for key in required_keys:
            if key not in metrics:
                logger.error(f"❌ metrics_experiments.json missing key: {key}")
                return False, metrics
        
        # Validate results
        results = metrics.get("results", {})
        if "rf" not in results or "gb" not in results:
            logger.error("❌ Results missing RF or GB benchmark")
            return False, metrics
        
        logger.info("✅ metrics_experiments.json: VALID")
        
        # Show metrics summary
        logger.info("\nMetrics Summary:")
        for model_name, model_metrics in results.items():
            logger.info(f"  {model_name.upper()}:")
            logger.info(f"    - RMSE: {model_metrics.get('RMSE', 'N/A'):.4f}")
            logger.info(f"    - MAE:  {model_metrics.get('MAE', 'N/A'):.4f}")
            logger.info(f"    - MAPE: {model_metrics.get('MAPE', 'N/A'):.4f}")
            logger.info(f"    - Clarke A+B: {model_metrics.get('Clarke_A+B', 'N/A'):.2f}%")
        
        return True, metrics
    
    except json.JSONDecodeError as e:
        logger.error(f"❌ metrics_experiments.json: INVALID JSON ({str(e)})")
        return False, None
    except Exception as e:
        logger.error(f"❌ metrics_experiments.json: ERROR ({str(e)})")
        return False, None


def verify_data_source():
    """Verify that OhioT1DM data exists."""
    logger.info("=" * 70)
    logger.info("DATA SOURCE VERIFICATION")
    logger.info("=" * 70)
    
    data_file = DATA_DIR / "ohio_t1dm_merged.csv"
    
    if not data_file.exists():
        logger.warning(f"⚠️  {data_file.name}: NOT FOUND (may use generated data)")
        return False
    
    try:
        df = pd.read_csv(data_file, nrows=1)
        num_rows = len(pd.read_csv(data_file))
        logger.info(f"✅ {data_file.name}: FOUND ({num_rows} rows)")
        return True
    except Exception as e:
        logger.error(f"❌ {data_file.name}: FAILED TO READ ({str(e)})")
        return False


def create_lock_manifest(models_ok, metrics_ok, metrics):
    """Create a lock manifest documenting the artifact state."""
    logger.info("=" * 70)
    logger.info("ARTIFACT LOCK MANIFEST")
    logger.info("=" * 70)
    
    manifest = {
        "lock_timestamp": datetime.utcnow().isoformat(),
        "status": "LOCKED" if (models_ok and metrics_ok) else "INCOMPLETE",
        "verification": {
            "models_verified": models_ok,
            "metrics_verified": metrics_ok,
            "data_verified": False,  # Set in main
        },
        "artifacts": {
            "models": {
                "rf_baseline.pkl": (MODELS_DIR / "rf_baseline.pkl").exists(),
                "gb_baseline.pkl": (MODELS_DIR / "gb_baseline.pkl").exists(),
            },
            "metrics_file": "metrics_experiments.json",
        },
        "metrics_summary": None,
    }
    
    if metrics and "results" in metrics:
        manifest["metrics_summary"] = metrics["results"]
    
    manifest_path = MODELS_DIR / "ARTIFACTS_LOCK_MANIFEST.json"
    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"✅ Lock manifest saved: {manifest_path.name}")
        logger.info(f"   Status: {manifest['status']}")
    except Exception as e:
        logger.error(f"❌ Failed to save lock manifest: {str(e)}")
    
    return manifest


def main():
    logger.info("\n" + "=" * 70)
    logger.info("DAY 5: PREDICTION ARTIFACT LOCK VERIFICATION")
    logger.info("=" * 70 + "\n")
    
    # Run verifications
    models_ok = verify_model_artifacts()
    metrics_ok, metrics = verify_metrics_file()
    data_ok = verify_data_source()
    
    # Create lock manifest
    manifest = create_lock_manifest(models_ok, metrics_ok, metrics)
    
    # Final report
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Models verified:     {models_ok}")
    logger.info(f"Metrics verified:    {metrics_ok}")
    logger.info(f"Data source verified: {data_ok}")
    logger.info(f"\nOverall Status:      {manifest['status']}")
    logger.info("=" * 70 + "\n")
    
    if manifest["status"] == "LOCKED":
        logger.info("✅ All prediction artifacts successfully verified and locked!")
        logger.info("   Ready for Day 6: Twin state transition stabilization.")
        return 0
    else:
        logger.error("⚠️  Some artifacts could not be verified. Please review above.")
        return 1


if __name__ == "__main__":
    exit(main())
