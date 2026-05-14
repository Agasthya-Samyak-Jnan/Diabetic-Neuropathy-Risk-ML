import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import joblib
import cv2

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from model import HybridModel

# -----------------------------
# Load models
# -----------------------------
MODEL_PATH = "models/MobileNetV2-base-model-v02.keras"
XGB_PATH   = "models/xgb_hybrid_model.joblib"
SCALER_PATH = "models/scaler.joblib"

hybrid_model = HybridModel(MODEL_PATH)
xgb_model = joblib.load(XGB_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# Risk mapping
# -----------------------------
def risk_label(val):
    if val <= 0.5:
        return "LOW RISK"
    elif val <= 1.5:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"

def risk_color(val):
    if val <= 0.5:
        return "#4CAF50"
    elif val <= 1.5:
        return "#FFC107"
    else:
        return "#F44336"

# -----------------------------
# App window
# -----------------------------
root = tk.Tk()
root.title("Diabetic Neuropathy Risk Predictor")
root.geometry("750x850")
root.configure(bg="#f5f5f5")

selected_image_path = None

# Heatmap toggle state
is_heatmap = False
original_img_tk = None
heatmap_img_tk = None

# -----------------------------
# FUNCTIONS
# -----------------------------
def upload_image():
    global selected_image_path, original_img_tk, heatmap_img_tk, is_heatmap

    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )

    if file_path:
        selected_image_path = file_path
        is_heatmap = False

        img = Image.open(file_path)
        img.thumbnail((300, 300))
        original_img_tk = ImageTk.PhotoImage(img)

        image_label.config(image=original_img_tk)
        image_label.image = original_img_tk

        heatmap_img_tk = None
        heatmap_btn.config(text="See Heatmap")

        result_label.config(text="Image Loaded", fg="black")


def toggle_heatmap():
    global is_heatmap, heatmap_img_tk

    if selected_image_path is None:
        return

    is_heatmap = not is_heatmap

    if is_heatmap:
        if heatmap_img_tk is None:
            img = cv2.imread(selected_image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            heatmap = cv2.applyColorMap(gray.astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (300, 300))
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            img_pil = Image.fromarray(heatmap)
            heatmap_img_tk = ImageTk.PhotoImage(img_pil)

        image_label.config(image=heatmap_img_tk)
        image_label.image = heatmap_img_tk
        heatmap_btn.config(text="See Image")

    else:
        image_label.config(image=original_img_tk)
        image_label.image = original_img_tk
        heatmap_btn.config(text="See Heatmap")


def show_graph(pred):
    fig, ax = plt.subplots(figsize=(5, 2.5))

    categories = ["Low", "Medium", "High"]
    values = [2.0, 2.0, 2.0]
    colors = ["green", "orange", "red"]

    ax.set_xticks([])
    ax.set_xticklabels([])

    plt.margins(0)

    # Horizontal bars
    ax.barh(categories, values, color=colors, alpha=0.3)

    # Vertical line for prediction
    ax.axhline(pred, color="blue", linestyle="--", label=f"Score: {pred:.2f}")

    ax.set_xlim(0, 2)
    ax.set_xlabel("Risk Score")
    ax.set_title("Risk Distribution")
    ax.legend()

    for widget in graph_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()


def predict_risk():
    if selected_image_path is None:
        messagebox.showwarning("Warning", "Upload image first")
        return

    try:
        features = hybrid_model.predict(selected_image_path)
        features_scaled = scaler.transform([features])

        pred = xgb_model.predict(features_scaled)[0]

        # Result
        result_label.config(
            text=f"{risk_label(pred)}\nScore: {pred:.2f}",
            fg=risk_color(pred),
            font=("Arial", 14, "bold")
        )

        # Progress bar
        progress["value"] = (pred / 2.0) * 100

        # Graph
        show_graph(pred)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# -----------------------------
# UI LAYOUT
# -----------------------------
title = tk.Label(
    root,
    text="🦶 Diabetic Neuropathy Risk Predictor",
    font=("Arial", 18, "bold"),
    bg="#f5f5f5"
)
title.pack(pady=10)

# Image
image_label = tk.Label(root, bg="#ddd")
image_label.pack(pady=10)

# Buttons
btn_frame = tk.Frame(root, bg="#f5f5f5")
btn_frame.pack(pady=5)

upload_btn = tk.Button(btn_frame, text="Upload", command=upload_image, width=15)
upload_btn.grid(row=0, column=0, padx=5)

heatmap_btn = tk.Button(btn_frame, text="See Heatmap", command=toggle_heatmap, width=15)
heatmap_btn.grid(row=0, column=1, padx=5)

predict_btn = tk.Button(
    btn_frame,
    text="Predict",
    command=predict_risk,
    bg="#4CAF50",
    fg="white",
    width=15
)
predict_btn.grid(row=0, column=2, padx=5)

# Result
result_label = tk.Label(
    root,
    text="Upload image to begin",
    font=("Arial", 12),
    bg="#f5f5f5"
)
result_label.pack(pady=10)

# Progress bar
progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress.pack(pady=10)

# Graph
graph_frame = tk.Frame(root, bg="#f5f5f5")
graph_frame.pack(pady=10)

# Run
root.mainloop()