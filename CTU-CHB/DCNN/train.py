import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import json
import pickle
import tensorflow as tf

from sklearn.utils import class_weight
from tensorflow import reshape
from tensorflow.keras import metrics
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.backend import batch_get_value
from tensorflow_model_optimization.sparsity import keras as sparsity


np.random.seed(42)
tf.random.set_seed(42)

with gzip.open('artifacts/train_images.npy.gz', 'rb') as f:
    train_images = np.load(f)

with gzip.open('artifacts/test_images.npy.gz', 'rb') as f:
    test_images = np.load(f)

train_labels = np.load('artifacts/train_labels.npy')
test_labels = np.load('artifacts/test_labels.npy')

train_images_resampled = train_images
train_labels_resampled = train_labels

# oversampler = RandomOverSampler()
# train_images_resampled = reshape(train_images_resampled, (train_images_resampled.shape[0], -1))
# train_images_resampled, train_labels_resampled = oversampler.fit_resample(train_images_resampled, train_labels.argmax(axis=1))
# train_images_resampled = train_images_resampled.reshape(train_images_resampled.shape[0], 150, 150, 3)
# train_labels_resampled = to_categorical(train_labels_resampled, num_classes=2)

class_weights = class_weight.compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(train_labels_resampled.argmax(axis=1)),
    y = train_labels_resampled.argmax(axis=1)
    )
class_weight_dict = dict(enumerate(class_weights))

model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    layers.Dropout(0.8),
    layers.Dense(2, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)
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

model.fit(
    train_images_resampled,
    train_labels_resampled,
    epochs=10,
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

model_json = model.to_json()
with open('model_architecture.json', 'w') as json_file:
    json.dump(model_json, json_file)

model.save_weights('model.weights.h5')

# optimizer is NOT saved!