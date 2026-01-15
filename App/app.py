import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
import os
from model import CNNmodel as MODEL
from model import extract_global_thermal_features

model = MODEL(
    model_path="models/MobileNetV2-base-model-v02.keras"
)

dropped_image_path = None

def drop_image(event):
    global dropped_image_path

    dropped_image_path = event.data.strip("{}")

    if os.path.isfile(dropped_image_path):
        status_label.config(text="Image loaded")
        print("Thermal image stored at:", dropped_image_path)
    else:
        status_label.config(text="Invalid file")
        dropped_image_path = None

def predict_image():
    if dropped_image_path is None:
        status_label.config(text="No image dropped ❗")
        return

    # ---- CNN Prediction ----
    prob, label = model.predict(dropped_image_path)

    # ---- Thermal Features ----
    features = extract_global_thermal_features(dropped_image_path)

    # ---- Format output ----
    output_text = (
        f"CNN Probability: {prob:.4f} | "
        f"Prediction: {'High Risk' if label else 'Low Risk'}"
    )

    status_label.config(text=output_text)

    # ---- Console Output (detailed) ----
    print("\n===== CNN OUTPUT =====")
    print(f"Probability: {prob:.4f}")
    print(f"Prediction : {'High Risk' if label else 'Low Risk'}")

    print("\n===== THERMAL FEATURES =====")
    for k, v in features.items():
        print(f"{k:25s}: {v:.4f}")

# Output = {prob,features} are everything we need for building weighted risk score

root = TkinterDnD.Tk()
root.title("Thermal CNN Predictor")
root.geometry("450x250")

label = tk.Label(
    root,
    text="Drag & Drop Thermal Image Here",
    bg="lightgray",
    width=45,
    height=6
)
label.pack(pady=10)

label.drop_target_register(DND_FILES)
label.dnd_bind("<<Drop>>", drop_image)

# Predict button
predict_btn = tk.Button(
    root,
    text="Predict CNN Probability",
    command=predict_image,
    width=25
)
predict_btn.pack(pady=10)

# Status / output label
status_label = tk.Label(root, text="Waiting for image...", fg="blue")
status_label.pack(pady=5)

root.mainloop()

