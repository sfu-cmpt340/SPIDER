import pandas as pd
pd.options.mode.chained_assignment = None

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score

from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt


# Load Dataset
pd.set_option('display.max_columns', None)
df = pd.read_csv('fetal_health.csv')

# Select columns for ML methods
df = df[['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations',
        'severe_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability',
        'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability',
        'mean_value_of_long_term_variability', 'fetal_health']]

# Create X and Y data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Check number of each class
normal_count = (df['fetal_health'] == 1).sum()
suspect_count = (df['fetal_health'] == 2).sum()
pathological_count = (df['fetal_health'] == 3).sum()
print("Normal Count: ", normal_count, " Suspec Count: ", suspect_count, " Pathological Count: ", pathological_count)

# Output the accuracies of each ML method
def PrintResults(X, y, model):
    # Split the data into 75% training and 25% testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Output the accuracies
    print(model)
    print(classification_report(y_test, y_pred))
    print("Balanced Accuracy: ", balanced_accuracy_score(y_test, y_pred))

    # Output the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[1,2,3])
    ConfusionMatrixDisplay(cm).plot()
    print("")
    
# Check which variable has the most importance for each ML method
def ShowFeatureImportance(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = model.fit(X_train, y_train)

    hist_df = pd.DataFrame({'Feature': X.columns, 'Feature importance': model.feature_importances_})
    hist_df = hist_df.sort_values(by='Feature importance', ascending=True)
    plt.figure(figsize=(10, 4))
    plt.barh(hist_df['Feature'], hist_df['Feature importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(model)
    plt.tight_layout()
    plt.show()

# DecisionTreeClassifier
model_dt = DecisionTreeClassifier()
PrintResults(X, y, model_dt)
ShowFeatureImportance(X, y, model_dt)

# GaussianNB
model_nb = GaussianNB()
PrintResults(X, y, model_nb)

X_train, X_test, y_train, y_test = train_test_split(X, y)
model_nb = model_nb.fit(X_train, y_train)
imps_gb = permutation_importance(model_nb, X_train, y_train)
plt.figure(figsize=(10, 4))
hist_bay = pd.DataFrame({'Feature': X.columns, 'Feature importance': imps_gb.importances_mean})
hist_bay = hist_bay.sort_values(by='Feature importance', ascending=True)
plt.barh(hist_bay['Feature'], hist_bay['Feature importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(model_nb)
plt.tight_layout()
plt.show()

# RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=200)
PrintResults(X, y, model_rf)
ShowFeatureImportance(X, y, model_rf)

# KNeighborsClassifier
model_knn = KNeighborsClassifier(2)
PrintResults(X, y, model_knn)
