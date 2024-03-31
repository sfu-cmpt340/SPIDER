import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gzip

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

labels = np.load('artifacts/labels.npy', allow_pickle=True)
filenames = np.load('artifacts/filenames.npy', allow_pickle=True)

print("Generating test/train split...")

images = []
for filename in filenames:
    try:
        path = os.path.join(os.getcwd(), '../spectrogram', filename)
        img = load_img(path, target_size=(150, 150))
        img_array = img_to_array(img)
        images.append(img_array)
    except Exception as e:
        print(f"Error loading image {filename}: {e}")

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.25, random_state=42)

train_images = tf.convert_to_tensor(train_images)
test_images = tf.convert_to_tensor(test_images)

with gzip.open('artifacts/train_images.npy.gz', 'wb') as f:
    np.save(f, train_images)

with gzip.open('artifacts/test_images.npy.gz', 'wb') as f:
    np.save(f, test_images)

np.save('artifacts/train_labels.npy', train_labels)
np.save('artifacts/test_labels.npy', test_labels)