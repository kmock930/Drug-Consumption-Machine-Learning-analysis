# self_training.py

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from joblib import dump


def load_split_data_pickle(directory='split_data_pickle'):
    """
    Loads the split and processed data saved as pickle files.
    """
    filenames = ['X_train.pkl', 'X_test.pkl', 'y_train.pkl', 'y_test.pkl', 'scaler.pkl']
    loaded_data = {}
    for filename in filenames:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                loaded_data[filename] = pickle.load(file)
            print(f"Loaded '{filename}' from '{directory}/'")
        else:
            raise FileNotFoundError(f"'{filename}' not found in '{directory}/'")
    return loaded_data['X_train.pkl'], loaded_data['X_test.pkl'], loaded_data['y_train.pkl'], loaded_data['y_test.pkl'], \
    loaded_data['scaler.pkl']


def save_model(model, directory='models', filename='self_training_model.joblib'):
    """
    Saves the trained model using Joblib.
    """
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, filename)
    dump(model, model_path)
    print(f"Model saved to '{model_path}'")


def build_self_training_model(X_train, X_test, y_train, y_test):
    """
    Builds and evaluates a Self-Training model.
    """
    # Initialize base classifier
    base_clf = GradientBoostingClassifier(random_state=42)

    # Initialize Self-Training Classifier
    self_training_clf = SelfTrainingClassifier(base_estimator=base_clf, threshold=0.8, max_iter=10, verbose=True)

    # Fit the model
    print("\nTraining Self-Training Classifier...")
    self_training_clf.fit(X_train, y_train)

    # Save the model
    save_model(self_training_clf, directory='models', filename='self_training_model.joblib')

    # Predictions
    y_pred = self_training_clf.predict(X_test)
    y_pred_proba = self_training_clf.predict_proba(X_test)[:, 1]

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Self-Training Test Accuracy: {accuracy:.4f}")
    print(f"Self-Training AUC-ROC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Self-Training Confusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Self-Training ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def main():
    # Load data
    X_train, X_test, y_train, y_test, scaler = load_split_data_pickle()

    # Build and evaluate the model
    build_self_training_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
