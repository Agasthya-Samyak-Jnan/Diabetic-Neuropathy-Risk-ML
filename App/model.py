import tensorflow as tf
import numpy as np
import os
import cv2
import numpy as np
from scipy.stats import skew
from skimage.measure import shannon_entropy

class CNNmodel:
    def __init__(self, model_path, img_size=224):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found")

        self.model = tf.keras.models.load_model(model_path)
        self.img_size = img_size

        print("Model loaded successfully")

    def preprocess(self, image_path):
        
        img = tf.keras.utils.load_img(
            image_path,
            target_size=(self.img_size, self.img_size)
        )

        img = tf.keras.utils.img_to_array(img)  # float32, 0-255
        img = tf.expand_dims(img, axis=0)       # (1, 224, 224, 3)

        return img

    def predict(self, image_path, threshold=0.5):
        """
        Returns probability and binary decision
        """
        img = self.preprocess(image_path)
        prob = self.model.predict(img, verbose=0)[0][0]
        label = int(prob >= threshold)

        return prob, label

def extract_global_thermal_features(image_path):
    # -----------------------------
    # Load image
    # -----------------------------
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or invalid format")

    img = img.astype(np.float32)

    # -----------------------------
    # Basic statistics
    # -----------------------------
    mean_temp = np.mean(img)
    max_temp  = np.max(img)
    min_temp  = np.min(img)
    std_temp  = np.std(img)
    temp_range = max_temp - min_temp

    # -----------------------------
    # Distribution features
    # -----------------------------
    skewness = skew(img.flatten())
    entropy  = shannon_entropy(img)

    # -----------------------------
    # Hot / Cold spot analysis
    # -----------------------------
    hotspot_thresh = mean_temp + 2 * std_temp
    coldspot_thresh = mean_temp - 2 * std_temp

    hotspot_fraction = np.sum(img > hotspot_thresh) / img.size
    coldspot_fraction = np.sum(img < coldspot_thresh) / img.size

    # -----------------------------
    # Left–Right asymmetry
    # -----------------------------
    h, w = img.shape
    left  = img[:, :w//2]
    right = img[:, w//2:]

    lr_mean_asymmetry = abs(np.mean(left) - np.mean(right))
    lr_std_asymmetry  = abs(np.std(left)  - np.std(right))

    # -----------------------------
    # Center–Periphery difference
    # -----------------------------
    center_mask = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(center_mask, (w//2, h//2), min(h, w)//4, 1, -1)

    center_region = img[center_mask == 1]
    periphery_region = img[center_mask == 0]

    center_periphery_diff = abs(np.mean(center_region) - np.mean(periphery_region))

    # -----------------------------
    # Pack features
    # -----------------------------
    features = {
        "mean_temp": mean_temp,
        "max_temp": max_temp,
        "min_temp": min_temp,
        "std_temp": std_temp,
        "temp_range": temp_range,
        "skewness": skewness,
        "entropy": entropy,
        "hotspot_fraction": hotspot_fraction,
        "coldspot_fraction": coldspot_fraction,
        "lr_mean_asymmetry": lr_mean_asymmetry,
        "lr_std_asymmetry": lr_std_asymmetry,
        "center_periphery_diff": center_periphery_diff
    }

    return features