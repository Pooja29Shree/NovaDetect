import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from PIL import Image

# Optional: Allow large image processing
Image.MAX_IMAGE_PIXELS = None

# ---------------------------
# Config
# ---------------------------
IMG_SIZE = 128
IMG_CHANNELS = 1
IMAGE_DIR = "images/"
MASK_DIR = "masks/"

# ---------------------------
# Load and preprocess images/masks
# ---------------------------
import re

def load_images(img_dir, mask_dir, img_size):
    image_list = []
    mask_list = []

    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.tif'))]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.tif'))]

    for img_file in image_files:
        # Extract timestamp using regex
        match = re.search(r'\d{8}T\d+', img_file) # e.g., 20090706T2122023762
        if not match:
            print(f"⚠ Skipping {img_file} — no timestamp found")
            continue

        timestamp = match.group(0)

        # Find matching mask
        matching_mask_file = None
        for mask_file in mask_files:
            if timestamp in mask_file:
                matching_mask_file = mask_file
                break

        if matching_mask_file is None:
            print(f"⚠ No matching mask for {img_file} (timestamp: {timestamp})")
            continue

        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, matching_mask_file)

        img = load_img(img_path, color_mode='grayscale', target_size=(img_size, img_size))
        mask = load_img(mask_path, color_mode='grayscale', target_size=(img_size, img_size))

        img_array = img_to_array(img) / 255.0
        mask_array = img_to_array(mask)
        mask_array = (mask_array > 127).astype(np.float32)

        image_list.append(img_array)
        mask_list.append(mask_array)

    print(f"✅ Loaded {len(image_list)} matched image-mask pairs.")
    return np.array(image_list), np.array(mask_list)



images, masks = load_images(IMAGE_DIR, MASK_DIR, IMG_SIZE)
print("Images shape:", images.shape)
print("Masks shape:", masks.shape)

# ---------------------------
# U-Net Architecture
# ---------------------------
def unet_model(input_size=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    # Bottleneck
    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)

    # Decoder
    u1 = UpSampling2D()(c3)
    m1 = concatenate([u1, c2])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(m1)
    c4 = Conv2D(32, 3, activation='relu', padding='same')(c4)

    u2 = UpSampling2D()(c4)
    m2 = concatenate([u2, c1])
    c5 = Conv2D(16, 3, activation='relu', padding='same')(m2)
    c5 = Conv2D(16, 3, activation='relu', padding='same')(c5)

    outputs = Conv2D(1, 1, activation='sigmoid')(c5)

    return Model(inputs, outputs)

# ---------------------------
# Dice Loss Function
# ---------------------------
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# ---------------------------
# Train/Test Split
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# ---------------------------
# Compile & Train
# ---------------------------
model = unet_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=dice_loss,
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=1)

# ---------------------------
# Predict and Visualize
# ---------------------------
pred_mask = model.predict(np.expand_dims(X_val[0], axis=0))[0]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(X_val[0].squeeze(), cmap='gray')
axes[0].set_title("Input Image")
axes[0].axis('off')

axes[1].imshow(y_val[0].squeeze(), cmap='gray')
axes[1].set_title("True Mask")
axes[1].axis('off')

axes[2].imshow(pred_mask.squeeze(), cmap='gray')
axes[2].set_title("Predicted Mask")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("prediction_output.png")

model.save("model.h5")
