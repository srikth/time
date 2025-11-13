data/
  train/
    angry/
    disgust/
    fear/
    happy/
    sad/
    surprise/
    neutral/
  val/
    angry/
    ...
# emotion_cnn.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


DATA_DIR = "data"         # should contain 'train' and 'val' subfolders
IMG_SIZE = (48, 48)       # common size for emotion datasets (FER)
BATCH_SIZE = 64
EPOCHS = 40
AUTOTUNE = tf.data.AUTOTUNE
MODEL_SAVE_PATH = "emotion_cnn.h5"

# emotion labels (common 7-class FER mapping)
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def make_datasets(data_dir=DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        labels="inferred",
        label_mode="categorical",   # one-hot
        color_mode="grayscale",     # facial emotion datasets are often grayscale
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=123
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        labels="inferred",
        label_mode="categorical",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    )

    # Prefetch for performance
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
])

def build_model(input_shape=(48,48,1), num_classes=7):
    inputs = layers.Input(shape=input_shape)

    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)

    # Block 1
    x = layers.Conv2D(32, (3,3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3,3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3,3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, (3,3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3,3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3,3), padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="EmotionCNN")
    return model


def train():
    train_ds, val_ds = make_datasets()

    # auto-detect input shape
    for batch, labels in train_ds.take(1):
        input_shape = batch.shape[1:]
    print("Input shape:", input_shape)

    model = build_model(input_shape=input_shape, num_classes=len(CLASS_NAMES))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Callbacks
    cb = [
        callbacks.ModelCheckpoint("best_emotion_cnn.h5", save_best_only=True, monitor="val_accuracy", mode="max"),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=cb
    )

    # Save final model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Plot training history (accuracy & loss)
    plot_history(history)

    return model, history


# Plot helper
def plot_history(history):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label="train acc")
    plt.plot(epochs, val_acc, label="val acc")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label="train loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.title("Loss")
    plt.legend()

    plt.show()

def load_trained_model(path=MODEL_SAVE_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    m = tf.keras.models.load_model(path, compile=False)
    return m

def predict_image(model, img_path, img_size=IMG_SIZE):
    img = image.load_img(img_path, color_mode="grayscale", target_size=img_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  # batch
    arr = arr / 255.0
    preds = model.predict(arr)
    class_idx = np.argmax(preds, axis=1)[0]
    prob = preds[0, class_idx]
    label = CLASS_NAMES[class_idx]
    return label, float(prob), preds[0]


# Optional: real-time webcam demo (requires opencv)

def webcam_demo(model, img_size=IMG_SIZE):
    try:
        import cv2
    except Exception as e:
        print("OpenCV not installed. Install with `pip install opencv-python` to use webcam demo.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # face detection (naive) — use Haar cascade shipped with OpenCV
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, img_size)
            arr = face.astype("float32") / 255.0
            arr = np.expand_dims(arr, axis=(0, -1))  # batch and channel dims
            preds = model.predict(arr)
            class_idx = np.argmax(preds, axis=1)[0]
            label = CLASS_NAMES[class_idx]
            prob = preds[0, class_idx]

            # draw
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            text = f"{label} {prob:.2f}"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        cv2.imshow("Emotion recognition (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Train model
    model, history = train()

    # Example prediction (adjust path)
    example_img = "example.jpg"
    if os.path.exists(example_img):
        label, prob, raw = predict_image(model, example_img)
        print(f"Predicted: {label} ({prob:.2f})")

    # Optionally run webcam demo:
    # webcam_demo(model)
