import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define file paths
base_path = r"C:\Users\suhas\OneDrive\Desktop\Natural Disaster Managemant\xBD Dataset\sample_xBD_sliced_augmented_20_alldisasters_final_mdl_npy"

train_pre_path = os.path.join(base_path, "train_pre_image_chips_0.npy")
train_post_path = os.path.join(base_path, "train_post_image_chips_0.npy")
train_mask_path = os.path.join(base_path, "train_dmg_mask_chips_0.npy")

# Ensure files exist
for path in [train_pre_path, train_post_path, train_mask_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

# Load dataset
train_pre_images = np.load(train_pre_path)
train_post_images = np.load(train_post_path)
train_damage_masks = np.load(train_mask_path)

# Print dataset shapes
print(f"Pre-disaster images shape: {train_pre_images.shape}")  # (N, 256, 256, 3)
print(f"Post-disaster images shape: {train_post_images.shape}")  # (N, 256, 256, 3)
print(f"Damage masks shape: {train_damage_masks.shape}")  # (N, 256, 256)

# Normalize image data (scale from 0-255 to 0-1)
train_pre_images = train_pre_images.astype("float32") / 255.0
train_post_images = train_post_images.astype("float32") / 255.0

# Check unique values in masks
unique_values = np.unique(train_damage_masks)
print("Unique values in damage masks before fixing:", unique_values)

# Fix mask values: Convert classes 4 & 5 into class 3 (Severe Damage)
train_damage_masks[train_damage_masks > 3] = 3

# Check unique values again
unique_values_fixed = np.unique(train_damage_masks)
print("Unique values in damage masks after fixing:", unique_values_fixed)

# Convert damage masks to categorical (one-hot encoding)
num_classes = 4
train_damage_masks = tf.keras.utils.to_categorical(train_damage_masks, num_classes=num_classes).astype("float32")

# Split into training and validation sets
X_train_pre, X_val_pre, X_train_post, X_val_post, y_train, y_val = train_test_split(
    train_pre_images, train_post_images, train_damage_masks, test_size=0.2, random_state=42
)

# Print train/validation split info
print(f"Training set: {X_train_pre.shape}, {X_train_post.shape}, {y_train.shape}")
print(f"Validation set: {X_val_pre.shape}, {X_val_post.shape}, {y_val.shape}")

# Function to visualize images
def show_sample(index):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(X_train_pre[index])
    axes[0].set_title("Pre-Disaster Image")
    axes[0].axis("off")

    axes[1].imshow(X_train_post[index])
    axes[1].set_title("Post-Disaster Image")
    axes[1].axis("off")

    axes[2].imshow(np.argmax(y_train[index], axis=-1), cmap="jet")  # Use jet colormap for better visibility
    axes[2].set_title("Damage Mask")
    axes[2].axis("off")

    plt.show()

# Display a sample image
show_sample(0)
# Save preprocessed data as .npy files
np.save("X_train_pre.npy", X_train_pre)
np.save("X_train_post.npy", X_train_post)
np.save("y_train.npy", y_train)

np.save("X_val_pre.npy", X_val_pre)
np.save("X_val_post.npy", X_val_post)
np.save("y_val.npy", y_val)

print("✅ Dataset saved successfully!")
