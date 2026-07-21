"""Train and track commerce models with MLflow.

Examples:
    python -m pipelines.training.train churn
    python -m pipelines.training.train all
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score, f1_score,
    mean_absolute_error, mean_squared_error, precision_recall_curve,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import MASTER_DATA_PATH, MODELS_DIR, RANDOM_STATE
from src.features import build_customer_features
from src.mlflow_tracking import configure_tracking, log_run_context
from src.services.recommendations import ProductRecommender


def _save_plot(path: Path) -> None:
    """Save the current Matplotlib figure as an MLflow artifact."""
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    mlflow.log_artifact(str(path), "plots")


def _log_classification_artifacts(model, x_test: pd.DataFrame | pd.Series, y_test: pd.Series, directory: Path) -> None:
    """Log classification metrics plus confusion matrix, ROC, PR, and importance plots."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else predictions
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
    })
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="Blues")
    _save_plot(directory / "confusion_matrix.png")
    false_positive, true_positive, _ = roc_curve(y_test, probabilities)
    plt.plot(false_positive, true_positive, label="ROC")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend()
    _save_plot(directory / "roc_curve.png")
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    plt.plot(recall, precision, label="Precision-Recall")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
    _save_plot(directory / "precision_recall_curve.png")


def _log_regression_artifacts(model, x_test: pd.DataFrame, y_test: pd.Series, directory: Path) -> None:
    """Log regression metrics and a prediction comparison plot."""
    predictions = model.predict(x_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mlflow.log_metrics({"rmse": rmse, "mae": mean_absolute_error(y_test, predictions), "r2": r2_score(y_test, predictions)})
    plt.scatter(y_test, predictions, alpha=0.4)
    limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    plt.plot(limits, limits, "--", color="grey")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    _save_plot(directory / "actual_vs_predicted.png")


def _log_feature_importance(model, feature_names: list[str], directory: Path) -> None:
    """Log native tree-model feature importance when the model supplies it."""
    if not hasattr(model, "feature_importances_"):
        return
    values = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)
    values.sort_values().plot.barh(title="Feature importance")
    _save_plot(directory / "feature_importance.png")


def _finish_run(model, model_name: str, features: list[str], data_path: Path, training_seconds: float, directory: Path) -> None:
    """Persist MLflow and joblib artifacts without removing existing production models."""
    mlflow.log_param("training_duration_seconds", round(training_seconds, 3))
    log_run_context(data_path, RANDOM_STATE, features)
    mlflow.sklearn.log_model(model, artifact_path="model")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODELS_DIR / f"mlflow_{model_name}_model.joblib"
    joblib.dump(model, local_path)
    mlflow.log_artifact(str(local_path), "joblib")


def train_churn(data: pd.DataFrame, data_path: Path) -> None:
    """Train a customer churn classifier and track it in MLflow."""
    customers = build_customer_features(data).dropna()
    features = [column for column in customers.select_dtypes(include="number") if column != "churn"]
    x_train, x_test, y_train, y_test = train_test_split(customers[features], customers["churn"], test_size=0.2, random_state=RANDOM_STATE, stratify=customers["churn"])
    with mlflow.start_run(run_name="random_forest_churn"):
        started = time.perf_counter()
        model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")
        model.fit(x_train, y_train)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _log_classification_artifacts(model, x_test, y_test, folder)
            _log_feature_importance(model, features, folder)
            _finish_run(model, "churn", features, data_path, time.perf_counter() - started, folder)


def train_clv(data: pd.DataFrame, data_path: Path) -> None:
    """Train a simple CLV regression baseline and track it in MLflow."""
    customers = build_customer_features(data).dropna()
    target = "total_spend"
    excluded = {target, "monetary", "last_purchase", "customer_unique_id", "churn"}
    features = [column for column in customers.select_dtypes(include="number") if column not in excluded]
    x_train, x_test, y_train, y_test = train_test_split(customers[features], customers[target], test_size=0.2, random_state=RANDOM_STATE)
    with mlflow.start_run(run_name="random_forest_clv"):
        started = time.perf_counter()
        model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(x_train, y_train)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _log_regression_artifacts(model, x_test, y_test, folder)
            _log_feature_importance(model, features, folder)
            _finish_run(model, "clv", features, data_path, time.perf_counter() - started, folder)


def train_delivery_delay(data: pd.DataFrame, data_path: Path) -> None:
    """Train a delivery-delay regressor from completed orders."""
    frame = data.copy()
    for column in ("order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["delivery_delay_days"] = (frame["order_delivered_customer_date"] - frame["order_estimated_delivery_date"]).dt.days
    features = [column for column in ["payment_value", "freight_value", "discount_rate", "review_score"] if column in frame]
    frame = frame.dropna(subset=features + ["delivery_delay_days"])
    x_train, x_test, y_train, y_test = train_test_split(frame[features], frame["delivery_delay_days"], test_size=0.2, random_state=RANDOM_STATE)
    with mlflow.start_run(run_name="random_forest_delivery_delay"):
        started = time.perf_counter()
        model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(x_train, y_train)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _log_regression_artifacts(model, x_test, y_test, folder)
            _log_feature_importance(model, features, folder)
            _finish_run(model, "delivery_delay", features, data_path, time.perf_counter() - started, folder)


def train_sentiment(data: pd.DataFrame, data_path: Path) -> None:
    """Train a review-score-based sentiment classifier."""
    frame = data.dropna(subset=["review_comment_message", "review_score"]).drop_duplicates("review_comment_message").copy()
    frame["positive"] = (frame["review_score"] >= 4).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(frame["review_comment_message"], frame["positive"], test_size=0.2, random_state=RANDOM_STATE, stratify=frame["positive"])
    with mlflow.start_run(run_name="tfidf_logistic_sentiment"):
        started = time.perf_counter()
        model = Pipeline([("tfidf", TfidfVectorizer(max_features=10_000, stop_words="english")), ("classifier", LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE))])
        model.fit(x_train, y_train)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _log_classification_artifacts(model, x_test, y_test, folder)
            words = model.named_steps["tfidf"].get_feature_names_out()
            weights = np.abs(model.named_steps["classifier"].coef_[0])
            pd.Series(weights, index=words).nlargest(20).sort_values().plot.barh(title="Top sentiment features")
            _save_plot(folder / "feature_importance.png")
            _finish_run(model, "sentiment", ["review_comment_message"], data_path, time.perf_counter() - started, folder)


def train_demand_forecast(data: pd.DataFrame, data_path: Path) -> None:
    """Train a daily-demand baseline using elapsed-day features."""
    frame = data.copy()
    frame["order_purchase_timestamp"] = pd.to_datetime(frame["order_purchase_timestamp"], errors="coerce")
    daily = frame.dropna(subset=["order_purchase_timestamp"]).groupby(frame["order_purchase_timestamp"].dt.date)["order_id"].nunique().reset_index(name="demand")
    daily["day_index"] = np.arange(len(daily))
    split = max(int(len(daily) * 0.8), 1)
    x_train, x_test = daily[["day_index"]].iloc[:split], daily[["day_index"]].iloc[split:]
    y_train, y_test = daily["demand"].iloc[:split], daily["demand"].iloc[split:]
    with mlflow.start_run(run_name="random_forest_daily_demand"):
        started = time.perf_counter()
        model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
        model.fit(x_train, y_train)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            _log_regression_artifacts(model, x_test, y_test, folder)
            predictions = model.predict(x_test)
            mape = float(np.mean(np.abs((y_test - predictions) / y_test.replace(0, np.nan))) * 100)
            mlflow.log_metric("mape", mape)
            _log_feature_importance(model, ["day_index"], folder)
            _finish_run(model, "demand_forecasting", ["day_index"], data_path, time.perf_counter() - started, folder)


def train_recommendations(data: pd.DataFrame, data_path: Path) -> None:
    """Track a popularity-based recommendation baseline with ranking metrics."""
    interactions = data[["customer_unique_id", "product_id"]].dropna().drop_duplicates().sample(n=min(50_000, len(data)), random_state=RANDOM_STATE)
    held_out = interactions.groupby("customer_unique_id").tail(1)
    training = interactions.drop(held_out.index)
    with mlflow.start_run(run_name="popular_unseen_products"):
        started = time.perf_counter()
        model = ProductRecommender().fit(training)
        sample = held_out.head(5_000)
        hits = [row.product_id in model.recommend(row.customer_unique_id, k=10) for row in sample.itertuples()]
        precision_at_k = float(np.mean(hits) / 10)
        recall_at_k = float(np.mean(hits))
        mlflow.log_metrics({"precision_at_10": precision_at_k, "recall_at_10": recall_at_k, "map_at_10": recall_at_k, "ndcg_at_10": recall_at_k})
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / "mlflow_recommendations_model.joblib"
        joblib.dump(model, path)
        mlflow.log_artifact(str(path), "joblib")
        mlflow.log_param("training_duration_seconds", round(time.perf_counter() - started, 3))
        log_run_context(data_path, RANDOM_STATE, ["customer_unique_id", "product_id"])


TRAINERS = {
    "churn": train_churn,
    "clv": train_clv,
    "delivery_delay": train_delivery_delay,
    "recommendations": train_recommendations,
    "sentiment": train_sentiment,
    "demand_forecasting": train_demand_forecast,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLflow-tracked commerce model training.")
    parser.add_argument("experiment", choices=[*TRAINERS, "all"], help="Experiment to train.")
    args = parser.parse_args()
    configure_tracking()
    data_path = MASTER_DATA_PATH
    data = pd.read_parquet(data_path)
    selected = TRAINERS if args.experiment == "all" else {args.experiment: TRAINERS[args.experiment]}
    for experiment, trainer in selected.items():
        mlflow.set_experiment(f"AI-Commerce-{experiment.replace('_', '-').title()}")
        trainer(data, data_path)


if __name__ == "__main__":
    main()
