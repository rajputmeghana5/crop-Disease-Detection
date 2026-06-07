import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


layers = tf.keras.layers
models = tf.keras.models
callbacks = tf.keras.callbacks
preprocessing = tf.keras.preprocessing.image


ImageDataGenerator = preprocessing.ImageDataGenerator
EarlyStopping = callbacks.EarlyStopping
ModelCheckpoint = callbacks.ModelCheckpoint
# =========================
# CONFIG
# =========================
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 12

train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =========================
# DATA GENERATOR (Normalization + Augmentation)
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,   # NORMALIZATION
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    shear_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical',
    shuffle=False
)

num_classes = len(train_data.class_indices)
class_names = list(train_data.class_indices.keys())

# =========================
# CLASS WEIGHTS (IMBALANCE FIX)
# =========================
labels = train_data.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

# =========================
# MODEL (Balanced CNN)
# =========================
model = models.Sequential([

    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(256, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# CALLBACKS
# =========================
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
  ModelCheckpoint("model/best_model.h5", save_best_only=True) 
]

# =========================
# TRAIN
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# =========================
# SAVE FINAL MODEL
# =========================
model.save("model/final_model.h5")

# =========================
# EVALUATION
# =========================
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(test_data.classes, y_pred_classes)

# =========================
# 12 IMPORTANT GRAPHS
# =========================

# 1 Accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title("Accuracy")
plt.savefig("outputs/accuracy.png")
plt.clf()

# 2 Loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.title("Loss")
plt.savefig("outputs/loss.png")
plt.clf()

# 3 Confusion Matrix
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.clf()

# 4 Class Distribution
unique, counts = np.unique(labels, return_counts=True)
plt.bar(unique, counts)
plt.title("Class Distribution")
plt.savefig("outputs/class_distribution.png")
plt.clf()

# 5 Prediction Confidence
plt.hist(np.max(y_pred, axis=1))
plt.title("Confidence Distribution")
plt.savefig("outputs/confidence.png")
plt.clf()

# 6 Learning Rate (static)
plt.plot([0.0001]*len(history.history['loss']))
plt.title("Learning Rate")
plt.savefig("outputs/lr.png")
plt.clf()

# 7 Precision/Recall/F1
report = classification_report(test_data.classes, y_pred_classes, output_dict=True)
precision = [report[str(i)]['precision'] for i in range(len(class_names))]
recall = [report[str(i)]['recall'] for i in range(len(class_names))]

plt.plot(precision, label='precision')
plt.plot(recall, label='recall')
plt.legend()
plt.title("Precision vs Recall")
plt.savefig("outputs/precision_recall.png")
plt.clf()

# 8 F1 Score
f1 = [report[str(i)]['f1-score'] for i in range(len(class_names))]
plt.plot(f1)
plt.title("F1 Score")
plt.savefig("outputs/f1.png")
plt.clf()

# 9 Epoch vs Accuracy
plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'])
plt.title("Epoch vs Accuracy")
plt.savefig("outputs/epoch_acc.png")
plt.clf()

# 10 Epoch vs Loss
plt.plot(range(len(history.history['loss'])), history.history['loss'])
plt.title("Epoch vs Loss")
plt.savefig("outputs/epoch_loss.png")
plt.clf()

# 11 Class Weights Graph
plt.bar(class_weights.keys(), class_weights.values())
plt.title("Class Weights")
plt.savefig("outputs/class_weights.png")
plt.clf()

# 12 Sample Predictions
plt.imshow(test_data[0][0][0])
plt.title("Sample Image")
plt.savefig("outputs/sample.png")
plt.clf()

print(" Training + Graphs Completed")