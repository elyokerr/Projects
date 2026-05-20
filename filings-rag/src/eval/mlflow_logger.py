"""MLflow logging for Ragas eval runs, tagged by retrieval configuration."""

import mlflow


def log_eval_run(
    metrics: dict,
    config: dict,
    experiment: str = "filings-rag-eval",
) -> str:
    """Log a single eval run to MLflow and return the run_id.

    `config` is intended for retrieval-config params (dense_only, bm25_only,
    hybrid, hybrid+rerank, +query_rewrite, etc.) so ablation runs can be
    compared in the MLflow UI.
    """
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(config)
        mlflow.log_metrics(metrics)
        return run.info.run_id
