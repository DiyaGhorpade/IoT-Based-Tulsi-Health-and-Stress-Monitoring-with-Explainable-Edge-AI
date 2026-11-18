#!/usr/bin/env python3
"""Local Streamlit dashboard for saved ML model evaluation and report screenshots.

Run from the money_plant_ml_pipeline directory with:
    streamlit run scripts/performance_dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SRC = PIPELINE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from plant_ml.config import load_config
from plant_ml.evaluation_run import evaluate_run


st.set_page_config(page_title="Model Performance Dashboard", page_icon="🌿", layout="wide")
st.title("Plant Stress Model — Performance Dashboard")
st.caption(
    "Evaluation uses the saved model and the recorded held-out split. "
    "The bundled dataset and labels are synthetic; results are not validated plant-health diagnoses."
)

DEFAULT_CSV = PIPELINE_ROOT / "data" / "raw" / "plant_sensor_data.csv"


@st.cache_data(show_spinner=False)
def run_evaluation(csv_path: str, run_id: str, with_shap: bool) -> dict:
    cfg = load_config()
    return evaluate_run(cfg, csv_path, run_id or None, with_shap=with_shap)


with st.sidebar:
    st.header("Evaluation settings")
    csv_path = st.text_input("Labeled CSV", value=str(DEFAULT_CSV))
    run_id = st.text_input("Artifact run ID (blank = latest)", value="")
    with_shap = st.checkbox("Include SHAP importance (may take longer)", value=False)
    refresh = st.button("Re-evaluate model", type="primary")
    if refresh:
        run_evaluation.clear()

try:
    input_path = Path(csv_path).expanduser().resolve()
    if not input_path.is_file():
        st.error(f"CSV file not found: {input_path}")
        st.stop()

    with st.spinner("Evaluating the saved model on the held-out split…"):
        result = run_evaluation(str(input_path), run_id.strip(), with_shap)
except Exception as exc:  # show actionable errors in the local dashboard
    st.error(f"Evaluation could not be completed: {exc}")
    st.info("Check that the CSV has the required labels and that trained artifacts exist under artifacts/.")
    st.stop()

classifier = result["classifier"]
unsmoothed = result.get("classifier_unsmoothed", {})
provenance = result.get("provenance", "unknown")

st.subheader("Evaluation overview")
st.write(f"**Artifact run:** `{result['run_id']}`  ·  **Data provenance:** `{provenance}`  ·  **Samples:** {classifier['n_samples']}")
if result.get("warning"):
    st.warning(result["warning"])

metric_cols = st.columns(4)
for col, label, key in zip(
    metric_cols,
    ("Accuracy", "Macro F1", "Weighted F1", "Balanced accuracy"),
    ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"),
):
    col.metric(label, f"{classifier[key]:.3f}")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    cm_data = classifier["confusion_matrix"]
    cm = np.asarray(cm_data["matrix"])
    labels = cm_data["labels"]
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Held-out test set")
    fig.colorbar(image, ax=ax, label="Number of readings")
    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center",
                    color="white" if cm[row, col] > threshold else "black")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    st.subheader("Per-class precision, recall & F1")
    per_class = classifier["per_class"]
    class_names = list(per_class)
    scores = {metric: [per_class[name][metric] for name in class_names]
              for metric in ("precision", "recall", "f1")}
    x = np.arange(len(class_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7, 5))
    for index, (metric, values) in enumerate(scores.items()):
        ax.bar(x + (index - 1) * width, values, width, label=metric.title())
    ax.set_xticks(x, class_names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Performance by stress class")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Smoothed vs. unsmoothed classifier")
    compare_keys = ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")
    if unsmoothed:
        compare_names = ("Accuracy", "Macro F1", "Weighted F1", "Balanced accuracy")
        x = np.arange(len(compare_keys))
        width = 0.36
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(x - width / 2, [classifier[k] for k in compare_keys], width, label="Smoothed")
        ax.bar(x + width / 2, [unsmoothed[k] for k in compare_keys], width, label="Unsmoothed")
        ax.set_xticks(x, compare_names, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No unsmoothed evaluation metrics are available for this artifact.")

with right:
    st.subheader("Test-set class distribution")
    counts = classifier["class_counts"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(list(counts), list(counts.values()), color="#4b8b62")
    ax.set_ylabel("Number of readings")
    ax.set_title("True labels in held-out test set")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.divider()
st.subheader("Other model checks")
anomaly = result.get("anomaly")
cluster = result.get("clustering") or {}
check_cols = st.columns(5)
if anomaly:
    for col, label, key in zip(check_cols[:4], ("Anomaly precision", "Anomaly recall", "Anomaly F1", "Anomaly PR-AUC"),
                               ("precision", "recall", "f1", "pr_auc")):
        col.metric(label, f"{anomaly[key]:.3f}")
    check_cols[4].metric("Cluster silhouette", f"{cluster['test_silhouette']:.3f}" if cluster.get("test_silhouette") is not None else "N/A")
    st.caption(anomaly.get("note", "Anomaly results depend on the supplied helper labels."))
else:
    st.info("Anomaly metrics are unavailable (the evaluation data needs both helper-label classes).")
    st.metric("Cluster silhouette", f"{cluster['test_silhouette']:.3f}" if cluster.get("test_silhouette") is not None else "N/A")

st.success(f"Evaluation files and plots saved to: {result['reports_dir']}")
if result.get("shap"):
    if result["shap"].get("available"):
        shap_plot = Path(result["reports_dir"]) / "shap_global_importance.png"
        if shap_plot.is_file():
            st.subheader("Global feature importance (SHAP)")
            st.image(str(shap_plot), use_container_width=True)
    else:
        st.warning(f"SHAP was unavailable: {result['shap'].get('reason', 'optional dependency not installed')}")

st.caption("For report screenshots, use your browser's print-to-PDF or screenshot tool. Metrics describe the held-out synthetic evaluation data only.")
