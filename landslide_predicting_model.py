import os
import re
import cv2
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ----------------------------
# Config
# ----------------------------
IMG_SIZE = 128
MODEL_PATH = "model.h5"  # <- your trained model
CSV_PATH = "landslides_output.csv"
OUTPUT_IMG = "predicted_output.png"

# ----------------------------
# Load model
# ----------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ----------------------------
# Predict + Save Mask + Export CSV
# ----------------------------
def process_image(image_path):
    # Load and preprocess image
    img = load_img(image_path, color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

    # Predict
    pred_mask = model.predict(img_input)[0]
    binary_mask = (pred_mask.squeeze() > 0.5).astype(np.uint8) * 255

    # Save prediction image
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_array.squeeze(), cmap='gray')
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.title("Predicted Mask (soft)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(binary_mask, cmap='gray')
    plt.title("Predicted Mask (binary)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"✅ Saved output image: {OUTPUT_IMG}")

    # Extract timestamp from filename
    match = re.search(r'\d{8}T\d+', image_path)
    timestamp = match.group(0) if match else "unknown"

    # Extract contours from binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    landslides = []
    for contour in contours:
        if cv2.contourArea(contour) < 5:  # Filter small areas
            continue

        x, y, w, h = cv2.boundingRect(contour)
        landslides.append({
            "timestamp": timestamp,
            "geometry_type": "bbox",
            "geometry": f"x:{x}, y:{y}, w:{w}, h:{h}"
        })

    # Save to CSV
    df = pd.DataFrame(landslides)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Saved {len(df)} landslide(s) to CSV: {CSV_PATH}")


# ----------------------------
# Run
# ----------------------------
if _name_ == "_main_":
    if len(sys.argv) != 2:
        print("Usage: python predict_single_and_export.py path/to/image.png")
    else:
        process_image(sys.argv[1])
