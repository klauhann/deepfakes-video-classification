import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

labels_path = Path("test_vids_label_small.csv")

st.set_page_config(page_title="Deepfake Video Classification", layout="wide")
st.title("Deepfake Video Classification")
try:
    df = pd.read_csv(labels_path)
    st.subheader("Test Videos Labels")
    st.dataframe(df, width="content")
except FileNotFoundError:
    st.error(labels_path, "not found.")

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
        ]
        result = subprocess.run(command, capture_output=True, text=True)

    plots_dir = Path("explanation")
    plot_files = sorted(plots_dir.glob("timeline_plot_*.png"))
    video_map = {Path(p).name: p for p in df["vids_list"].tolist()} if "df" in locals() else {}

    preds_path = Path("Y_predictions.npy")
    probs_path = Path("Y_probabilities.npy")
    test_data = pd.read_csv(labels_path) if labels_path.exists() else None
    y_predictions = np.load(preds_path) if preds_path.exists() else None
    y_probabilities = np.load(probs_path) if probs_path.exists() else None
    true_labels = test_data["label"].values if test_data is not None else None
    vids_list = test_data["vids_list"].tolist() if test_data is not None else []
    vid_to_index = {vid: i for i, vid in enumerate(vids_list)}
    label_text = {0: "Real", 1: "Fake"}

    st.subheader("Results")
    if plot_files:
        for plot_file in plot_files:
            raw_id = plot_file.stem.replace("timeline_plot_", "")
            video_id = Path(raw_id).stem
            video_filename = f"{video_id}.mp4"
            video_path = video_map.get(video_filename)
            idx = None
            if video_filename in vid_to_index:
                idx = vid_to_index[video_filename]
            elif f"test/{video_filename}" in vid_to_index:
                idx = vid_to_index[f"test/{video_filename}"]

            pred_label = int(y_predictions[idx]) if idx is not None else None
            true_label = int(true_labels[idx]) if idx is not None else None
            is_correct = pred_label == true_label if idx is not None else None

            st.subheader(str(idx) + ": " + video_filename)

            col_video, col_plot = st.columns(2, vertical_alignment="center")
            with col_video:
                if video_path and Path(video_path).exists():
                    st.video(video_path, width="stretch")
                else:
                    st.info(f"Video nicht gefunden: {video_filename}")
            with col_plot:
                st.image(str(plot_file), caption=plot_file.name, width="content")

          
            if idx is not None and y_predictions is not None and true_labels is not None:
                prob_text = ""
                if y_probabilities is not None:
                    prob = float(y_probabilities[idx]) if np.ndim(y_probabilities) == 1 else float(np.max(y_probabilities[idx]))
                    prob_text = f"Gesamtscore: {prob:.4f} | "

                st.write(
                    f"{prob_text}Vorhersage: {label_text.get(pred_label, pred_label)} | "
                    f"Label: {label_text.get(true_label, true_label)} | "
                    f"Richtig: {'Ja' if is_correct else 'Nein'}"
                )
            else:
                st.info("Keine Vorhersage für dieses Video gefunden.")
           
            frames_dir = Path("explanation/frames") / video_id
            frame_files = sorted(frames_dir.glob("frame_*.png"))
            if frame_files:
                st.caption(f"Frames for {video_id}")
                cols = st.columns(10)
                for idx, frame_file in enumerate(frame_files):
                    col = cols[idx % 10]
                    frame_id = frame_file.stem.replace("frame_", "")
                    col.image(str(frame_file), caption=frame_id, width=100)
            else:
                st.info(f"Keine Frames gefunden für {video_id}.")
    else:
        st.info("No timeline plots found in explanation/.")


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
