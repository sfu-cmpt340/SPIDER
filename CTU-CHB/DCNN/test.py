import pandas as pd
import numpy as np
import os
import gzip
import json
import tensorflow as tf

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

with open('model_architecture.json', 'r') as json_file:
    architecture = json.load(json_file)
model = tf.keras.models.model_from_json(architecture)
model.load_weights('model.weights.h5')

with gzip.open('artifacts/test_images.npy.gz', 'rb') as f:
    test_images = np.load(f)
test_labels = np.load('artifacts/test_labels.npy')

predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)
class_counts = np.bincount(predicted_classes)

print("Count of guesses in each class:")
for cls, count in enumerate(class_counts):
    print(f"Class {cls}: {count} guesses")

accuracy = accuracy_score(true_classes, predicted_classes)
recall = recall_score(true_classes, predicted_classes, average='macro')
f1 = f1_score(true_classes, predicted_classes, average='macro')
precision = precision_score(true_classes, predicted_classes, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"Precision: {precision}")