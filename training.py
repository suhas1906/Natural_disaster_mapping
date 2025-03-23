import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load preprocessed data
X_train_pre = np.load("X_train_pre.npy")
X_train_post = np.load("X_train_post.npy")
y_train = np.load("y_train.npy")

X_val_pre = np.load("X_val_pre.npy")
X_val_post = np.load("X_val_post.npy")
y_val = np.load("y_val.npy")

# Ensure data type optimization for memory efficiency
X_train_pre = X_train_pre.astype("float16")
X_train_post = X_train_post.astype("float16")
y_train = y_train.astype("float16")

X_val_pre = X_val_pre.astype("float16")
X_val_post = X_val_post.astype("float16")
y_val = y_val.astype("float16")

print("✅ Data loaded and converted to float16")

# Define a simple CNN model
def build_model():
    input_pre = keras.Input(shape=(256, 256, 3), name="pre_disaster")
    input_post = keras.Input(shape=(256, 256, 3), name="post_disaster")

    # Concatenate pre- and post-disaster images
    x = layers.Concatenate()([input_pre, input_post])

    # Use a simple CNN-based U-Net-like segmentation model
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

    # Decoder (upsampling back to 256x256)
    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same")(x)
    
    # Output layer with softmax activation for multi-class segmentation
    output = layers.Conv2D(4, (1, 1), activation="softmax")(x)  # Shape (256, 256, 4)

    model = keras.Model(inputs=[input_pre, input_post], outputs=output)

    return model


# Compile model
model = build_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    [X_train_pre, X_train_post], y_train,
    validation_data=([X_val_pre, X_val_post], y_val),
    batch_size=4,  # Reduce batch size if you get memory errors
    epochs=10
)

# Save trained model
model.save("damage_segmentation_model.h5")
print("✅ Model training completed and saved successfully!")

