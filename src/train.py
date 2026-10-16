import argparse
import json
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Ensure src directory is available in python path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess import clean_text


def load_imdb_data(samples: int | None = None, random_state: int = 42):
    """
    Loads the official IMDB train and test datasets from Hugging Face.
    Optionally subsamples data for smoke tests.
    """
    print("Loading IMDB dataset from Hugging Face...")
    dataset = load_dataset("stanfordnlp/imdb")

    train_texts = list(dataset["train"]["text"])
    train_labels = list(dataset["train"]["label"])
    test_texts = list(dataset["test"]["text"])
    test_labels = list(dataset["test"]["label"])

    if samples is not None and samples < len(train_texts):
        print(f"Subsampling {samples} examples for smoke testing...")
        train_texts, _, train_labels, _ = train_test_split(
            train_texts,
            train_labels,
            train_size=samples,
            stratify=train_labels,
            random_state=random_state,
        )
        test_size = min(samples, len(test_texts))
        test_texts, _, test_labels, _ = train_test_split(
            test_texts,
            test_labels,
            train_size=test_size,
            stratify=test_labels,
            random_state=random_state,
        )

    print(f"Dataset ready: {len(train_texts)} train samples, {len(test_texts)} test samples.")
    return train_texts, train_labels, test_texts, test_labels


def preprocess_corpus(texts: list[str]) -> list[str]:
    """
    Applies text cleaning to a list of raw text reviews.
    """
    print(f"Cleaning {len(texts)} reviews...")
    return [clean_text(t) for t in texts]


def build_pipeline(random_state: int = 42) -> Pipeline:
    """
    Builds the scikit-learn Pipeline combining TF-IDF and Logistic Regression.
    - No stop_words removal: preserves sentiment-critical negations like 'not', 'no', 'never'.
    - N-gram range (1, 2): captures phrases like 'not good' alongside single words.
    - LogisticRegression: provides fast, interpretable probability estimates.
    """
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words=None,  # Intentionally kept to retain negation context
    )
    classifier = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=random_state,
    )
    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


def evaluate_model(pipeline: Pipeline, X_test: list[str], y_test: list[int], y_train: list[int]):
    """
    Evaluates model performance and computes metrics including a majority-class baseline.
    """
    print("Evaluating model on test split...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # Majority-class baseline: always predict the most frequent class in train
    majority_class = int(pd.Series(y_train).mode()[0])
    baseline_acc = float(np.mean(np.array(y_test) == majority_class))

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="binary")), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="binary")), 4),
        "f1": round(float(f1_score(y_test, y_pred, average="binary")), 4),
        "majority_baseline_accuracy": round(baseline_acc, 4),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"],
        digits=4,
    )
    cm = confusion_matrix(y_test, y_pred)

    return metrics, report, cm, y_pred, y_prob


def export_misclassified(
    raw_texts: list[str],
    y_true: list[int],
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    limit: int = 10,
):
    """
    Exports real misclassified test examples for error analysis.
    """
    label_map = {0: "Negative", 1: "Positive"}
    misclassified_items = []

    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            misclassified_items.append({
                "index": i,
                "text": raw_texts[i],
                "true_label": label_map[y_true[i]],
                "predicted_label": label_map[y_pred[i]],
                "predicted_probability": round(float(y_prob[i][y_pred[i]]), 4),
                "positive_probability": round(float(y_prob[i][1]), 4),
            })
            if len(misclassified_items) >= limit:
                break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(misclassified_items, f, indent=2)

    print(f"Exported {len(misclassified_items)} misclassified examples to {output_path}")


def save_results(pipeline: Pipeline, metrics: dict, report: str, cm: np.ndarray):
    """
    Saves model checkpoint, metrics JSON, text report, and confusion matrix plot.
    """
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save trained pipeline
    model_path = models_dir / "tfidf_logreg.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")

    # 2. Save metrics json
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # 3. Save classification report text
    report_path = results_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved classification report to {report_path}")

    # 4. Save confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = results_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix chart to {cm_path}")


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression Sentiment Classifier")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Subsample train/test count for fast smoke testing (default: full dataset)",
    )
    args = parser.parse_args()

    # 1. Load data
    train_texts, y_train, test_texts, y_test = load_imdb_data(samples=args.samples, random_state=42)

    # 2. Preprocess text
    X_train = preprocess_corpus(train_texts)
    X_test = preprocess_corpus(test_texts)

    # 3. Build & fit pipeline
    print("Fitting TF-IDF + Logistic Regression pipeline...")
    pipeline = build_pipeline(random_state=42)
    pipeline.fit(X_train, y_train)

    # 4. Evaluate
    metrics, report, cm, y_pred, y_prob = evaluate_model(pipeline, X_test, y_test, y_train)
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy:                   {metrics['accuracy']:.4f}")
    print(f"Precision:                  {metrics['precision']:.4f}")
    print(f"Recall:                     {metrics['recall']:.4f}")
    print(f"F1-Score:                   {metrics['f1']:.4f}")
    print(f"Majority Baseline Accuracy: {metrics['majority_baseline_accuracy']:.4f}")
    print("\nClassification Report:\n", report)

    # 5. Save all outputs
    save_results(pipeline, metrics, report, cm)

    # 6. Export misclassified test instances for viva error analysis
    misclassified_path = PROJECT_ROOT / "results" / "misclassified.json"
    export_misclassified(test_texts, y_test, y_pred, y_prob, misclassified_path, limit=10)

    print("\nTraining and evaluation completed successfully!")


if __name__ == "__main__":
    main()
