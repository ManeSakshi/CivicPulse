import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)
import seaborn as sns
import os


def load_data():
    """
    Loads the labeled civic dataset from repo.
    Automatically picks: civic_labeled.csv or civicpulse_with_labels.csv
    """

    candidates = [
        "data/processed/civic_labeled.csv",
        "data/processed/civicpulse_with_labels.csv"
    ]

    for path in candidates:
        if os.path.exists(path):
            print(f"[INFO] Loaded dataset: {path}")
            return pd.read_csv(path)

    raise FileNotFoundError("❌ No labeled dataset found in data/processed/")


def preprocess(df):
    """
    Automatically detects the text column and sentiment column
    based on the actual dataset structure.
    """

    # Text column candidates
    text_candidates = [
        "clean_text", "processed_text", "text",
        "content", "final_text", "body"
    ]

    # Sentiment label candidates (your dataset uses 'label')
    sentiment_candidates = [
        "sentiment", "label", "sentiment_label",
        "vader_textblob_sentiment"
    ]

    # Detect text column
    text_col = None
    for col in text_candidates:
        if col in df.columns:
            text_col = col
            break

    # Detect sentiment column
    label_col = None
    for col in sentiment_candidates:
        if col in df.columns:
            label_col = col
            break

    if text_col is None:
        raise ValueError(f"❌ No text column found. Columns: {df.columns.tolist()}")

    if label_col is None:
        raise ValueError(f"❌ No sentiment label column found. Columns: {df.columns.tolist()}")

    print(f"[INFO] Using text column: {text_col}")
    print(f"[INFO] Using sentiment column: {label_col}")

    # Filter usable rows
    df = df.dropna(subset=[text_col, label_col])

    # Keep only positive / negative / neutral
    df = df[df[label_col].isin(["positive", "negative", "neutral"])]

    X = df[text_col].astype(str)
    y = df[label_col]

    return X, y

def train_and_evaluate(X, y):
    """
    Splits data, trains TF-IDF + Logistic Regression, evaluates metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=8000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n==================== MODEL PERFORMANCE ====================")
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, vectorizer, accuracy, report, y_test, y_pred


def save_metrics(accuracy, report, file_path="model_info.json"):
    """
    Saves accuracy + per-class metrics to a JSON file.
    """
    info = {
        "model": "TF-IDF + LogisticRegression",
        "accuracy": accuracy,
        "metrics": report
    }

    with open(file_path, "w") as f:
        json.dump(info, f, indent=4)

    print(f"[INFO] Saved metrics → {file_path}")


def save_confusion_matrix(y_true, y_pred):
    """
    Saves confusion matrix as PNG.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["negative", "neutral", "positive"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    out_path = "confusion_matrix.png"
    plt.savefig(out_path)
    plt.close()

    print(f"[INFO] Saved confusion matrix → {out_path}")


def main():
    print("[INFO] Starting sentiment model evaluation...\n")

    df = load_data()
    X, y = preprocess(df)
    model, vectorizer, acc, report, y_test, y_pred = train_and_evaluate(X, y)

    save_metrics(acc, report)
    save_confusion_matrix(y_test, y_pred)

    print("\n[COMPLETED] Evaluation finished successfully.")
    print("→ model_info.json created")
    print("→ confusion_matrix.png created")
    print("============================================================\n")


if __name__ == "__main__":
    main()
