import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# preprocessing
def normalize_column(df, column_name, min_val, max_val):
    df[column_name + '_normalized'] = (df[column_name].clip(lower=min_val) - min_val) / (max_val - min_val)
    return df

def check_hypoxia_conditions(row):
    # Check conditions for each column
    if (
        row['pH'] < 7.15
        or row['BDecf'] > 10
        or row['pCO2'] > 10
        or row['BE'] < -10
        or row['Apgar5'] < 7
        ):
        return 1
    else:
        return 0

normal_columns = {
    'pH': (7, 7.4),
    'BDecf': (-2, 20),
    'pCO2': (3, 10),
    'BE': (2, 10)
}

df = pd.read_csv('../outcomes.csv').dropna()

filenames = df["filename"]
df['hypoxia'] = df.apply(check_hypoxia_conditions, axis=1)
print(df['hypoxia'].value_counts())

labels = df['hypoxia']
labels = to_categorical(labels, num_classes=2)

np.save('artifacts/labels.npy', labels)
np.save('artifacts/filenames.npy', filenames)