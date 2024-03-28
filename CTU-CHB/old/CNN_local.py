import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import balanced_accuracy_score, classification_report, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

tf.test.is_built_with_cuda()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
rebuild_train_test = False

if rebuild_train_test:
    labels_df = pd.read_csv('outcomes.csv')
    filenames = labels_df["filename"]
    labels = labels_df["Apgar1"]
    images = []
    for filename in filenames:
        try:
            path = os.path.join(os.getcwd(), 'spectrogram', filename)
            img = tf.keras.preprocessing.image.load_img(path, target_size=(312, 312))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.25, random_state=42)

    train_images = tf.convert_to_tensor(train_images)
    test_images = tf.convert_to_tensor(test_images)
    print("Training images shape:", train_images.shape)
    print("Testing images shape:", test_images.shape)
    print("Training labels shape:", train_labels.shape)
    print("Testing labels shape:", test_labels.shape)
    train_images = tf.convert_to_tensor(train_images)
    test_images = tf.convert_to_tensor(test_images)
    np.save('train_images.npy', train_images)
    np.save('test_images.npy', test_images)

train_images = np.load('train_images.npy')
test_images = np.load('test_images.npy')

model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(312, 312, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax', kernel_regularizer=l2(0.001))

])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=1)

plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images, verbose=2)
