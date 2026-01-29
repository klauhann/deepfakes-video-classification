import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

LABELS_CSV = Path("test_vids_label.csv")
EXPLANATION_DIR = Path("explanation")
HEATMAPS_DIR = EXPLANATION_DIR / "heatmaps"
TIMELINE_PREFIX = "timeline_plot_"
LABEL_TEXT = {0: "Real", 1: "Fake"}

st.set_page_config(page_title="Deepfake Video Classification", layout="wide")
st.title("Deepfake Video Classification")
try:
    df = pd.read_csv(LABELS_CSV)
    st.subheader("Test Videos Labels")
    st.dataframe(df, width="content")
except FileNotFoundError:
    st.error(LABELS_CSV, "not found.")

if st.button("Classify Videos using CNN"):
    with st.spinner("Running evaluation..."):
        command = [
            sys.executable,
            "02.evaluate_CNN.py",
            "-m",
            "xception",
            "-w",
            "xception_best",
            "-im_size",
            "160",
            "-v",
            str(LABELS_CSV)
        ]
        result = subprocess.run(command, capture_output=True, text=True)

    plot_files = sorted(EXPLANATION_DIR.glob(f"{TIMELINE_PREFIX}*.png"))
    plot_map = {p.name: p for p in plot_files}

    preds_path = Path("Y_predictions.npy")
    probs_path = Path("Y_probabilities.npy")

    test_data = pd.read_csv(LABELS_CSV) if LABELS_CSV.exists() else None
    y_predictions = np.load(preds_path) if preds_path.exists() else None
    y_probabilities = np.load(probs_path) if probs_path.exists() else None
    true_labels = test_data["label"].values if test_data is not None else None
    vids_list = test_data["vids_list"].tolist() if test_data is not None else []

    st.header("Results")
    if vids_list:
        for idx, video in enumerate(vids_list):
            video_path = Path(video)
            video_filename = video_path.name
            video_id = Path(video_path.name).stem

            pred_label = int(y_predictions[idx]) if y_predictions is not None else None
            true_label = int(true_labels[idx]) if true_labels is not None else None
            is_correct = pred_label == true_label if pred_label is not None and true_label is not None else None

            st.subheader(f"{idx}: {video_filename}")

            if pred_label is not None and true_label is not None:
                prob_text = ""
                if y_probabilities is not None:
                    prob = float(y_probabilities[idx]) if np.ndim(y_probabilities) == 1 else float(np.max(y_probabilities[idx]))
                    prob_text = f"Gesamtscore: {prob:.4f} | "

                st.write(
                    f"{prob_text}Vorhersage: {LABEL_TEXT.get(pred_label, pred_label)} | "
                    f"Label: {LABEL_TEXT.get(true_label, true_label)} | "
                    f"Richtig: {'Ja' if is_correct else 'Nein'}"
                )
            else:
                st.info("Keine Vorhersage für dieses Video gefunden.")

            col_video, col_plot = st.columns(2, vertical_alignment="center")
            with col_video:
                if video_path.exists():
                    st.video(str(video_path), width="stretch")
                else:
                    st.info(f"Video nicht gefunden: {video}")
            with col_plot:
                plot_file = plot_map.get(f"{TIMELINE_PREFIX}{video_id}.png")
                if plot_file is not None and Path(plot_file).exists():
                    st.image(str(plot_file), caption=plot_file.name, width="content")
                else:
                    st.info("Kein Timeline-Plot vorhanden.")

            frames_dir = HEATMAPS_DIR / video_id
            frame_files = sorted(frames_dir.glob("frame_*.png"))
            if frame_files:
                st.caption(f"Frames for {video_id}")
                cols = st.columns(10)
                for frame_idx, frame_file in enumerate(frame_files):
                    col = cols[frame_idx % 10]
                    frame_id = frame_file.stem.replace("frame_", "")
                    col.image(str(frame_file), caption=frame_id, width=100)
            else:
                st.info(f"Keine Frames gefunden für {video_id}.")
    else:
        st.info("No videos found in the CSV.")


    if y_predictions is not None and true_labels is not None:
        st.subheader("Final Scores")
        st.write(
            {
                "Accuracy": accuracy_score(true_labels, y_predictions),
                "Precision": precision_score(true_labels, y_predictions),
                "Recall": recall_score(true_labels, y_predictions),
                "F1": f1_score(true_labels, y_predictions),
            }
        )
    else:
        st.info("Predictions not found yet. Run evaluation first.")
