import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from monitoring.metric_preset import DataDriftPreset
from monitoring.report import Report

logger = logging.getLogger(__name__)

# Mantemos o default, mas permitimos sobrescrever
DEFAULT_OUTPUT_PATH = Path("reports/drift_report.json")


def run_drift_detection(
    reference_path: str, current_path: str, save_path: Optional[str] = None
) -> dict:
    """Compara dados de treino vs produção e calcula drift."""

    # Define o caminho de saída (usa o argumento ou o padrão)
    output_file = Path(save_path) if save_path else DEFAULT_OUTPUT_PATH

    reference_df = pd.read_parquet(reference_path)
    current_df = pd.read_parquet(current_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    result = report.as_dict()

    # Extração segura do drift share
    drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "drift_share": drift_share,
        "n_reference": len(reference_df),
        "n_current": len(current_df),
    }

    # Garante que a pasta existe antes de salvar
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(
        "Drift detection executado: %.2f%% colunas com drift", drift_share * 100
    )

    return output
