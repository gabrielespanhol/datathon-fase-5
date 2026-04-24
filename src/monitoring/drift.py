# src/monitoring/drift.py
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("reports/drift_report.json")


def run_drift_detection(
    reference_path: str,
    current_path: str,
) -> dict:
    """Compara dados de treino vs produção e calcula drift."""

    reference_df = pd.read_parquet(reference_path)
    current_df = pd.read_parquet(current_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    result = report.as_dict()

    drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "drift_share": drift_share,
        "n_reference": len(reference_df),
        "n_current": len(current_df),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(
        "Drift detection executado: %.2f%% colunas com drift", drift_share * 100
    )

    return output
