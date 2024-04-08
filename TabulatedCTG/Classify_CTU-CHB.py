import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

# Load datasets Kaggle data and CTU-CHB data
pd.set_option('display.max_columns', None)
kaggle_df = pd.read_csv('fetal_health.csv')
CTU_df = pd.read_csv('../CTU-CHB/Feature Extraction/waveform_data.csv')

# Select columns for ML methods
kaggle_df = kaggle_df[['baseline value', 'accelerations', 'light_decelerations',
                        'prolongued_decelerations', 'abnormal_short_term_variability',
                        'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability',
                        'mean_value_of_long_term_variability', 'fetal_health']]
# Rename light_decelerations to decelerations
# There is no distinction between light/severe decelerations in the research papers
# Severe decelerations has a count of 0
kaggle_df = kaggle_df.rename(columns={'light_decelerations':'decelerations'})

# Create X and Y data
X = kaggle_df.iloc[:, :-1]
y = kaggle_df.iloc[:, -1]
# Split the data into 75% training and 25% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Declare all ML methods
model_dt = DecisionTreeClassifier()
model_nb = GaussianNB()
model_rf = RandomForestClassifier(n_estimators=200)
model_knn = KNeighborsClassifier(2)
models = [model_dt, model_nb, model_rf, model_knn]

# Output the prediction histograms for each method
for model in models:
    model.fit(X_train, y_train)
    pred = model.predict(CTU_df)
    plt.figure(figsize=(4, 4))
    bins = np.arange(0.5, 4.5, 1)  
    plt.hist(pred, bins=bins)
    plt.title(model)
    plt.xlabel("Predicted Class")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(1, 4, 1))
    plt.show()
