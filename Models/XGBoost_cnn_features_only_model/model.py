import tensorflow as tf
import numpy as np
import os
import cv2
from scipy.stats import skew
from skimage.measure import shannon_entropy

# ============================================================
# CNN BASED MODEL
# ============================================================

class CNNModel:
    def __init__(self, model_path, img_size=224):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found")

        # Load full Keras model
        full_model = tf.keras.models.load_model(model_path)

        # Cut model before final activation
        self.cnn_model = tf.keras.Model(
            inputs=full_model.input,
            outputs=full_model.layers[-2].output
        )

        self.img_size = img_size
        print("Hybrid CNN + Thermal feature extractor ready")
        print(f"🔹 CNN embedding dimension: {self.cnn_model.output_shape[-1]}")
        print("🔹 Thermal features dimension: 12")

    # -----------------------------
    # Preprocess CNN input
    # -----------------------------
    def preprocess(self, image_path):
        img = tf.keras.utils.load_img(
            image_path,
            target_size=(self.img_size, self.img_size)
        )
        img = tf.keras.utils.img_to_array(img)
        img = tf.expand_dims(img, axis=0)  # (1, H, W, 3)
        img = img / 255.0                  # normalize
        return img

    # -----------------------------
    # Extract CNN features
    # -----------------------------
    def extract_cnn_features(self, image_path):
        img = self.preprocess(image_path)
        embedding = self.cnn_model.predict(img, verbose=0)
        return embedding.flatten()  # (N,)

    # -----------------------------
    # Get deep CNN features
    # -----------------------------
    def predict(self, image_path):
        return self.extract_cnn_features(image_path)
