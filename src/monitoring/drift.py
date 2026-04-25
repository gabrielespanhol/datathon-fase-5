import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp  # Teste estatístico para comparar distribuições

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("reports/drift_report.json")


def run_drift_detection(
    reference_path: str,
    current_path: str,
    threshold: float = 0.05,  # P-value abaixo de 0.05 indica drift
) -> dict:
    """Compara dados de treino vs produção usando o teste Kolmogorov-Smirnov."""

    reference_df = pd.read_parquet(reference_path)
    current_df = pd.read_parquet(current_path)

    drifted_columns = 0
    total_columns = 0

    # Analisamos apenas colunas numéricas que existem em ambos os datasets
    cols_to_check = reference_df.select_dtypes(include=["number"]).columns

    for col in cols_to_check:
        if col in current_df.columns:
            total_columns += 1
            # O teste ks_2samp retorna (estatística, p-value)
            # Se p-value < threshold, as distribuições são significativamente diferentes
            _, p_value = ks_2samp(reference_df[col].dropna(), current_df[col].dropna())

            if p_value < threshold:
                drifted_columns += 1
                logger.warning(
                    f"Drift detectado na coluna: {col} (p-value: {p_value:.4f})"
                )

    drift_share = drifted_columns / total_columns if total_columns > 0 else 0

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drift_share": drift_share,
        "n_reference": len(reference_df),
        "n_current": len(current_df),
        "n_drifted_features": drifted_columns,
        "total_features_checked": total_columns,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(
        "Drift detection finalizado: %.2f%% colunas com drift", drift_share * 100
    )

    return output
