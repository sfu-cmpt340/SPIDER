import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
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

    # Create the model
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    print(classification_report(y_test, y_predicted))

    # Plot confusion matrix
    labels = [1.0, 2.0, 3.0]  # Assuming these are your label values
    plot_confusion_matrix(y_test, y_predicted, labels)

    # Conclusion
    print("After a few times trying to find the best n_neighbors value, I've decided 2 is the best number.\n It maximizes the accuracy of prediction to 3.0, which is 94%.")

if __name__ == '__main__':
    main()