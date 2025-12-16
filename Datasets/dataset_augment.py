import os
import cv2
import numpy as np

# ==================================
# CONFIG (Give TOP-LEVEL folder only)
# ==================================
ROOT = "Extended_Dataset"   # <<--- YOUR MAIN FOLDER HERE


# ==================================
# AUGMENT FUNCTIONS
# ==================================

def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def translate(img, percent):
    h, w = img.shape[:2]
    tx = int(w * percent)
    ty = int(h * percent)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def zoom(img, scale):
    h, w = img.shape[:2]
    resized = cv2.resize(img, None, fx=scale, fy=scale)
    if scale > 1:
        sx = (resized.shape[1] - w) // 2
        sy = (resized.shape[0] - h) // 2
        return resized[sy:sy+h, sx:sx+w]
    else:
        canvas = np.zeros_like(img)
        ox = (w - resized.shape[1]) // 2
        oy = (h - resized.shape[0]) // 2
        canvas[oy:oy+resized.shape[0], ox:ox+resized.shape[1]] = resized
        return canvas

def adjust_brightness(img, factor):
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def adjust_contrast(img, factor):
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def add_noise(img, sigma):
    noise = np.random.normal(0, sigma * 255, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def blur(img, sigma):
    return cv2.GaussianBlur(img, (3, 3), sigma)


# ==================================
# MAIN: RECURSIVE WALK
# ==================================

for root, dirs, files in os.walk(ROOT):
    for filename in files:

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        full_path = os.path.join(root, filename)
        img = cv2.imread(full_path)
        if img is None:
            print(f"Skipping unreadable: {filename}")
            continue

        base, ext = os.path.splitext(filename)

        # Skip if already augmented
        if any(tag in base for tag in 
               ["rotp8", "rotm8", "transp8", "transm8", "zoomin", "zoomout",
                "brightup", "brightdown", "contup", "contdown",
                "noise1", "noise3", "blur05", "blur10"]):
            continue

        print(f"Augmenting: {full_path}")

        # Save all augmentations directly in same folder
        cv2.imwrite(os.path.join(root, f"{base}_rotp8{ext}"), rotate(img, +6))
        cv2.imwrite(os.path.join(root, f"{base}_rotm8{ext}"), rotate(img, -6))

        cv2.imwrite(os.path.join(root, f"{base}_transp8{ext}"), translate(img, +0.08))
        cv2.imwrite(os.path.join(root, f"{base}_transm8{ext}"), translate(img, -0.08))

        cv2.imwrite(os.path.join(root, f"{base}_zoomin{ext}"), zoom(img, 1.10))
        cv2.imwrite(os.path.join(root, f"{base}_zoomout{ext}"), zoom(img, 0.90))

        cv2.imwrite(os.path.join(root, f"{base}_brightup{ext}"), adjust_brightness(img, 1.10))
        cv2.imwrite(os.path.join(root, f"{base}_brightdown{ext}"), adjust_brightness(img, 0.90))

        cv2.imwrite(os.path.join(root, f"{base}_contup{ext}"), adjust_contrast(img, 1.10))
        cv2.imwrite(os.path.join(root, f"{base}_contdown{ext}"), adjust_contrast(img, 0.90))

        cv2.imwrite(os.path.join(root, f"{base}_noise1{ext}"), add_noise(img, 0.01))
        cv2.imwrite(os.path.join(root, f"{base}_noise3{ext}"), add_noise(img, 0.03))

        cv2.imwrite(os.path.join(root, f"{base}_blur05{ext}"), blur(img, 0.5))
        cv2.imwrite(os.path.join(root, f"{base}_blur10{ext}"), blur(img, 1.0))

print("\nDone! Recursive augmentation complete.")
