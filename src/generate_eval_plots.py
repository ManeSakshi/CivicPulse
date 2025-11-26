import json
import matplotlib.pyplot as plt
import numpy as np
import os


def load_model_info(path="model_info.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model info file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def plot_per_class_metrics(model_info, out_path="per_class_metrics.png"):
    metrics = model_info.get("metrics", {})

    classes = ["negative", "neutral", "positive"]
    precision = [metrics[c]["precision"] for c in classes]
    recall = [metrics[c]["recall"] for c in classes]
    f1 = [metrics[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision, width, label="Precision", color="#4C72B0")
    plt.bar(x, recall, width, label="Recall", color="#55A868")
    plt.bar(x + width, f1, width, label="F1-score", color="#C44E52")

    plt.ylim(0, 1.0)
    plt.xticks(x, [c.capitalize() for c in classes])
    plt.ylabel("Score")
    plt.title("Per-class metrics: precision / recall / f1-score")
    plt.legend()

    # annotate bars
    for i in range(len(classes)):
        plt.text(x[i] - width, precision[i] + 0.01, f"{precision[i]:.2f}", ha="center", va="bottom", fontsize=9)
        plt.text(x[i], recall[i] + 0.01, f"{recall[i]:.2f}", ha="center", va="bottom", fontsize=9)
        plt.text(x[i] + width, f1[i] + 0.01, f"{f1[i]:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved per-class metrics plot → {out_path}")


def main():
    info = load_model_info("model_info.json")
    os.makedirs("reports", exist_ok=True)
    plot_per_class_metrics(info, out_path="per_class_metrics.png")


if __name__ == "__main__":
    main()
