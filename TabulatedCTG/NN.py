from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, classification_report, precision_score, recall_score, f1_score

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.utils import shuffle, class_weight
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

from scipy.stats import chi2_contingency
X = df.drop(columns='fetal_health')

contingency_tables = []
for column in X.columns:
    contingency_table = pd.crosstab(X[column], df['fetal_health'])
    contingency_tables.append((column, contingency_table))

results = []
for column, contingency_table in contingency_tables:
    chi2, p, _, _ = chi2_contingency(contingency_table)
    results.append((column, chi2, p))

results.sort(key=lambda x: x[1])

feature_names = [result[0] for result in results]
chi2v = [result[1] for result in results]

plt.barh(feature_names, chi2v)
plt.xlabel('chi2')
plt.ylabel('Feature')
plt.title('chi-square')
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import pandas as pd

df_filtered = df[df['fetal_health'].isin([1, 2])]
y = df_filtered['fetal_health']
df_filtered = df_filtered.drop(columns='fetal_health')

X_train, X_test, y_train, y_test = train_test_split(df_filtered, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

feature_importance_df = pd.DataFrame({'Feature': df_filtered.columns, 'Importance': perm_importance.importances_mean})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('importance')
plt.ylabel('Feature')
plt.title('feature importance between class 1 and 2')
plt.show()


pca = PCA()
pca.fit(df)

# Plotting explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance Ratio vs Number of Components')
plt.grid(True)
plt.show()

allfeatures = df.columns
# dropped = ['fetal_health', 'histogram_width', 'histogram_max', 'histogram_min', 'histogram_median', 'histogram_mean', 'histogram_mode', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_tendency']
dropped = [
    'fetal_health', 'baseline value', 'histogram_width',
    'light_decelerations', 'severe_decelerations', 'histogram_median',
    'histogram_mode', 'histogram_number_of_peaks',
    'histogram_number_of_zeroes', 'histogram_tendency', 'fetal_movement',
    'histogram_max', 'fetal_movement',
    ]

X = df.drop(columns=dropped)
y = df['fetal_health']-1

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# pca = PCA(n_components=6)
# X_train_pca = pca.fit_transform(X_train_scaled)
# X_test_pca = pca.transform(X_test_scaled)

# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
# class_weight_dict = dict(enumerate(class_weights))

reg_loss = 0.001
dropout = 0.1
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=l2(reg_loss)),
    BatchNormalization(),
    Dropout(dropout),
    Dense(256, activation='relu', kernel_regularizer=l2(reg_loss)),
    BatchNormalization(),
    Dropout(dropout),
    Dense(128, activation='relu', kernel_regularizer=l2(reg_loss)),
    BatchNormalization(),
    Dropout(dropout),
    Dense(64, activation='relu', kernel_regularizer=l2(reg_loss)),
    BatchNormalization(),
    Dropout(dropout),
    Dense(32, activation='relu', kernel_regularizer=l2(reg_loss)),
    BatchNormalization(),
    Dropout(dropout),
    Dense(3, activation='softmax')
])

sgd = SGD(learning_rate=0.01, momentum=0.9)
adam = Adam(learning_rate=0.001)
model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=64, validation_split=0.2, class_weight=class_weight_dict, verbose=0)
history = model.fit(X_train_scaled, y_train, epochs=300, batch_size=64, validation_split=0.25, verbose=0)

plt.plot(history.history['loss'], label='train Loss')
plt.plot(history.history['val_loss'], label='test Loss')
plt.legend()
plt.show()

train_accuracy = model.evaluate(X_train_scaled, y_train, verbose=0)[1]
test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)[1]

y_train_pred_nn = np.argmax(model.predict(X_train_scaled), axis=-1)
y_test_pred_nn = np.argmax(model.predict(X_test_scaled), axis=-1)

train_accuracy_nn = accuracy_score(y_train, y_train_pred_nn)
test_accuracy_nn = accuracy_score(y_test, y_test_pred_nn)
cm_nn = confusion_matrix(y_test, y_test_pred_nn)

ConfusionMatrixDisplay(cm_nn).plot()

print("Training accuracy:", train_accuracy_nn)
print("Test accurac:", test_accuracy_nn)

balanced_accuracy_train_nn = balanced_accuracy_score(y_train, y_train_pred_nn)
balanced_accuracy_test_nn = balanced_accuracy_score(y_test, y_test_pred_nn)

print("Train balanced acc:", balanced_accuracy_train_nn)
print("Test balanced acc:", balanced_accuracy_test_nn)
print("\nClassification Report:\n", classification_report(y_test, y_test_pred_nn))