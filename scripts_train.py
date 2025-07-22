import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# ─── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE  = (160, 160)          # input image size for MobileNetV2
BATCH     = 32                  # how many images per training batch
EPOCHS_H  = 5                   # epochs for head training
EPOCHS_F  = 5                   # epochs for fine-tuning
TRAIN_DIR = 'data/train'        # path to your training folders
VAL_DIR   = 'data/val'          # path to your validation folders
MODEL_DIR = 'models'            # where to save checkpoints

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Data Generators ──────────────────────────────────────────────────────────
# Train with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='binary'
)

# Validation (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='binary'
)

# ─── Build & Compile Model ─────────────────────────────────────────────────────
# 1) Load pre-trained base
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze all layers in the base

# 2) Add custom head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# 3) Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ─── Callbacks ──────────────────────────────────────────────────────────────────
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, 'best_head.h5'),
    save_best_only=True,
    monitor='val_accuracy'
)
earlystop = EarlyStopping(
    patience=3,
    restore_best_weights=True,
    monitor='val_accuracy'
)

# ─── Train Head ─────────────────────────────────────────────────────────────────
print("\n--- Training head (only custom layers) ---")
history_head = model.fit(
    train_gen,
    epochs=EPOCHS_H,
    validation_data=val_gen,
    callbacks=[checkpoint, earlystop]
)

# ─── Fine-tuning ────────────────────────────────────────────────────────────────
# Unfreeze last N layers of the base
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

checkpoint_f = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, 'best_finetune.h5'),
    save_best_only=True,
    monitor='val_accuracy'
)

print("\n--- Fine-tuning last layers of base model ---")
history_fine = model.fit(
    train_gen,
    epochs=EPOCHS_F,
    validation_data=val_gen,
    callbacks=[checkpoint_f, earlystop]
)

print("\nTraining complete! Best models saved in the `models/` folder.")
