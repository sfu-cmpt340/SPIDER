import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def main():

    # Get the dataframe
    df = pd.read_csv("fetal_health.csv")
    dropped = ['fetal_health', 'histogram_min', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency']

    # Split the dataframe for training data and validation data
    X = df.drop(columns=dropped)
    y = df["fetal_health"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get the priors of each fetal_health
    prior1 = df['fetal_health'].value_counts()[1.0] / 2126
    prior2 = df['fetal_health'].value_counts()[2.0] / 2126
    prior3 = df['fetal_health'].value_counts()[3.0] / 2126

    # Apply Bayesian Classifier
    model = GaussianNB(priors=[prior1, prior2, prior3])
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    print(classification_report(y_test, y_predicted))

    # Plot confusion matrix
    labels = [1.0, 2.0, 3.0]  # Assuming these are your label values
    plot_confusion_matrix(y_test, y_predicted, labels)

    # Conclusion
    print("This method is really good at predicting normal class data (1.0)\nWith accuracy of 94%, but it behaves poorly on the other classes.")
    
    
if __name__ == '__main__':
    main()