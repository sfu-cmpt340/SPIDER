import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import json
import pickle
import sys
import tensorflow as tf

from sklearn.utils import class_weight
from tensorflow import reshape
from tensorflow.keras import metrics
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import batch_get_value
from tensorflow_model_optimization.sparsity import keras as sparsity

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

num_epochs = int(sys.argv[1])

np.random.seed(42)
tf.random.set_seed(42)

with gzip.open('artifacts/train_images.npy.gz', 'rb') as f:
    train_images = np.load(f)

with gzip.open('artifacts/test_images.npy.gz', 'rb') as f:
    test_images = np.load(f)

train_labels = np.load('artifacts/train_labels.npy')
test_labels = np.load('artifacts/test_labels.npy')

X_train = train_images
y_train = train_labels

# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True,
#     fill_mode='nearest'
# )

# batch_size = 32
# oversample_factor = 2
# datagen.fit(train_images_resampled)

# def generate_oversampled_data(X, y, steps_per_epoch):
#     gen = datagen.flow(X, y, batch_size=batch_size)
#     for _ in range(steps_per_epoch):
#         X_batch, y_batch = next(gen)
#         yield X_batch, y_batch

# steps_per_epoch_train = int(np.ceil(len(train_images_resampled) / batch_size / oversample_factor))
# steps_per_epoch_val = int(np.ceil(len(train_labels_resampled) / batch_size / oversample_factor))
# train_generator = generate_oversampled_data(train_images_resampled, train_labels_resampled)
# val_generator = generate_oversampled_data(test_images, test_labels)

# oversampler = RandomOverSampler()
# train_images_resampled = reshape(train_images_resampled, (train_images_resampled.shape[0], -1))
# train_images_resampled, train_labels_resampled = oversampler.fit_resample(train_images_resampled, train_labels.argmax(axis=1))
# train_images_resampled = train_images_resampled.reshape(train_images_resampled.shape[0], 150, 150, 3)
# train_labels_resampled = to_categorical(train_labels_resampled, num_classes=2)

class_weights = class_weight.compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(y_train.argmax(axis=1)),
    y = y_train.argmax(axis=1)
    )
class_weight_dict = dict(enumerate(class_weights))

# y_train_indices = np.argmax(y_train, axis=1)
# class_counts = np.bincount(y_train_indices)
# print("Number of samples in class 0:", class_counts[0])
# print("Number of samples in class 1:", class_counts[1])

model = Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((3, 3)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((3, 3)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((3, 3)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    layers.Dropout(0.8),
    layers.Dense(2, activation='softmax')
])

optimizer = Adam(learning_rate=0.0001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        metrics.Precision(),
        metrics.Recall(),
        metrics.F1Score(),
    ],
)

history = model.fit(
    X_train,
    y_train,
    epochs=num_epochs,
    batch_size=32,
    validation_split=0.25,
    verbose=1,
    class_weight=class_weight_dict
)

# pruning testing
# pruning_params = {
#     'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.90, begin_step=0, end_step=1000)
# }

# model = sparsity.prune_low_magnitude(model, **pruning_params)

# plt.figure(figsize=(8, 2))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend()
# plt.show()

model_json = model.to_json()
with open('model_architecture.json', 'w') as json_file:
    json.dump(model_json, json_file)

model.save_weights('model.weights.h5')

# optimizer is NOT saved!