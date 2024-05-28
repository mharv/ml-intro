import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=data.target_names)

    print("Accuracy:", accuracy)
    # Accuracy measures the proportion of correctly classified instances among all instances in the dataset.
    # It gives an overall measure of the model's performance.
    # Mathematically, accuracy is calculated as the sum of true positives and true negatives divided by the sum of all instances.

    print("Classification Report:\n", report)
    # The classification report provides several metrics to evaluate the performance of a classification model.
    # Here's an explanation of each component:

    # Precision: Precision measures the proportion of true positive predictions among all positive predictions made by the model.
    # It tells us how many of the predicted positive instances are actually positive.

    # Recall (also known as Sensitivity or True Positive Rate): Recall measures the proportion of true positive predictions among all actual positive instances in the dataset.
    # It tells us how many of the actual positive instances were correctly predicted by the model.

    # F1-Score: The F1-score is the harmonic mean of precision and recall.
    # It provides a single metric that balances both precision and recall.

    # Support: Support is the number of actual occurrences of each class in the dataset.
    # It gives an indication of the number of samples in each class.

    # Accuracy: Accuracy measures the proportion of correctly classified instances among all instances in the dataset.
    # It gives an overall measure of the model's performance.


if __name__ == "__main__":
    main()