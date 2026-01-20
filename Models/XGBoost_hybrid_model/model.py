import tensorflow as tf
import numpy as np
import os
import cv2
from scipy.stats import skew
from skimage.measure import shannon_entropy

# ============================================================
# HYBRID MODEL: CNN + GLOBAL THERMAL FEATURES
# ============================================================

class HybridModel:
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
    # Extract thermal features
    # -----------------------------
    @staticmethod
    def extract_thermal_features(image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or invalid format")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Basic stats
        mean_temp = np.mean(gray)
        max_temp  = np.max(gray)
        min_temp  = np.min(gray)
        std_temp  = np.std(gray)
        temp_range = max_temp - min_temp

        # Distribution features
        skewness = skew(gray.flatten())
        entropy  = shannon_entropy(gray)

        # Hot / Cold spot
        hotspot_thresh = mean_temp + 2 * std_temp
        coldspot_thresh = mean_temp - 2 * std_temp
        hotspot_fraction = np.sum(gray > hotspot_thresh) / gray.size
        coldspot_fraction = np.sum(gray < coldspot_thresh) / gray.size

        # Left-right asymmetry
        h, w = gray.shape
        left = gray[:, :w//2]
        right = gray[:, w//2:]
        lr_mean_asymmetry = abs(np.mean(left) - np.mean(right))
        lr_std_asymmetry  = abs(np.std(left) - np.std(right))

        # Center-periphery difference
        center_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(center_mask, (w//2, h//2), min(h, w)//4, 1, -1)
        center_region = gray[center_mask==1]
        periphery_region = gray[center_mask==0]
        center_periphery_diff = abs(np.mean(center_region) - np.mean(periphery_region))

        features = np.array([
            mean_temp,
            max_temp,
            min_temp,
            std_temp,
            temp_range,
            skewness,
            entropy,
            hotspot_fraction,
            coldspot_fraction,
            lr_mean_asymmetry,
            lr_std_asymmetry,
            center_periphery_diff
        ], dtype=np.float32)

        return features

    # -----------------------------
    # Predict hybrid features
    # -----------------------------
    def predict(self, image_path):
        cnn_features = self.extract_cnn_features(image_path)
        thermal_features = self.extract_thermal_features(image_path)
        hybrid_features = np.concatenate([cnn_features, thermal_features])
        return hybrid_features
