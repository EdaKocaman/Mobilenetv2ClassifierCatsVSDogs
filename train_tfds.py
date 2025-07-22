import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE   = 160
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE
EPOCHS_H   = 5
EPOCHS_F   = 5
MODEL_DIR  = 'models'

os.makedirs(MODEL_DIR, exist_ok=True)

# Log key hyperparameters
def log_config():
    print(f"Config: IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}, "
          f"EPOCHS_HEAD={EPOCHS_H}, EPOCHS_FINE={EPOCHS_F}")

# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data():
    print("Downloading & preparing Cats vs. Dogs dataset…")
    return tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:]'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

# ─── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = image / 255.0
    return image, label

# ─── Build Dataset ────────────────────────────────────────────────────────────
def build_datasets(ds_train, ds_val):
    train_ds = (
        ds_train
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .shuffle(1000)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        ds_val
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    return train_ds, val_ds

# ─── Build Model ───────────────────────────────────────────────────────────────
def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model

# ─── Callbacks ─────────────────────────────────────────────────────────────────
def get_callbacks():
    cp_head = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'best_head.keras'),
        monitor='val_accuracy',
        save_best_only=True
    )
    cp_ft = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'best_finetune.keras'),
        monitor='val_accuracy',
        save_best_only=True
    )
    es = EarlyStopping(
        patience=3,
        restore_best_weights=True,
        monitor='val_accuracy'
    )
    return cp_head, cp_ft, es

# ─── Plotting ──────────────────────────────────────────────────────────────────
def plot_metrics(history_head, history_ft):
    acc      = history_head.history['accuracy'] + history_ft.history['accuracy']
    val_acc  = history_head.history['val_accuracy'] + history_ft.history['val_accuracy']
    loss     = history_head.history['loss'] + history_ft.history['loss']
    val_loss = history_head.history['val_loss'] + history_ft.history['val_loss']
    epochs   = range(1, len(acc) + 1)

    # Plot Accuracy
    plt.figure()
    plt.plot(epochs, acc, label='Train accuracy')
    plt.plot(epochs, val_acc, label='Val accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()

    # Plot Loss
    plt.figure()
    plt.plot(epochs, loss, label='Train loss')
    plt.plot(epochs, val_loss, label='Val loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    log_config()
    (ds_train, ds_val), _ = load_data()
    train_ds, val_ds = build_datasets(ds_train, ds_val)
    model, base_model = build_model()

    cp_head, cp_ft, es = get_callbacks()

    print("\n>>> Training custom head…")
    history_head = model.fit(
        train_ds,
        epochs=EPOCHS_H,
        validation_data=val_ds,
        callbacks=[cp_head, es]
    )

    print("\n>>> Fine-tuning last layers…")
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    history_ft = model.fit(
        train_ds,
        epochs=EPOCHS_F,
        validation_data=val_ds,
        callbacks=[cp_ft, es]
    )

    print("\n✅ Done! Check the final val_accuracy above.")
    plot_metrics(history_head, history_ft)