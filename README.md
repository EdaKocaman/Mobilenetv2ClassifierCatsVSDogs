# Cats vs. Dogs Classifier using MobileNetV2

This project implements a high-accuracy, lightweight image classification model that distinguishes between cats and dogs using transfer learning. Built with TensorFlow and MobileNetV2, the model achieves **97.9% validation accuracy** in under 1 hour on a MacBook M2.

---

## Project Overview

- **Goal**: Binary classification of cats and dogs from images.
- **Model**: MobileNetV2 (pre-trained on ImageNet) with a custom classification head.
- **Dataset**: Kaggle Dogs vs. Cats dataset (20,000+ images).
- **Training Hardware**: Apple M1 MacBook.
- **Accuracy Achieved**: ~97.9% validation accuracy.
- **Approach**: Two-phase training with data augmentation and dropout regularization.

---

## Model Architecture

```python
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

---

## Training Strategy

1. **Head-only training**  
   - Freeze all base model layers.  
   - Train custom classification head for 5 epochs.

2. **Fine-tuning**  
   - Unfreeze the last 20 layers of MobileNetV2.  
   - Train the full model with a reduced learning rate (1e-5) for another 5 epochs.

**Callbacks Used**:
- `ModelCheckpoint` to save best weights
- `EarlyStopping` to prevent overfitting

---

##  Results

- Validation Accuracy: **97.9%**
- Low validation loss (~0.06)
- Training Time: Less than 1 hour
- No significant overfitting observed thanks to augmentation and dropout

---

## Data Augmentation

The following transformations were applied to increase generalization:

- Random horizontal flips
- Random zooms
- Random shifts
- Random rotations

---

## References

- [Kaggle Dogs vs. Cats Dataset](https://www.kaggle.com/competitions/dogs-vs-cats)
- MobileNetV2: Sandler et al., 2018
- TensorFlow Documentation
- Oxford Pets Dataset (Parkhi et al., 2012)

---

## License

This project is licensed under the MIT License. See `LICENSE` file for details.
