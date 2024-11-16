
import pandas as pd
from sklearn.semi_supervised import LabelSpreading
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


def save_model(model, directory='models', filename='label_spreading_model.joblib'):
    """
    Saves the trained model using Joblib.
    """
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, filename)
    dump(model, model_path)
    print(f"Model saved to '{model_path}'")


def build_label_spreading_model(X_train, X_test, y_train, y_test):
    """
    Builds and evaluates a Label Spreading model.
    """
    # Initialize Label Spreading model
    label_spread = LabelSpreading(kernel='rbf', gamma=20, max_iter=30)

    # Fit the model
    print("\nTraining Label Spreading Classifier...")
    label_spread.fit(X_train, y_train)

    # Save the model
    save_model(label_spread, directory='models', filename='label_spreading_model.joblib')

    # Predictions
    y_pred = label_spread.predict(X_test)
    y_pred_proba = label_spread.predict_proba(X_test)[:, 1]

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Label Spreading Test Accuracy: {accuracy:.4f}")
    print(f"Label Spreading AUC-ROC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Label Spreading Confusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='purple', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Label Spreading ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def main():
    # Load data
    X_train, X_test, y_train, y_test, scaler = load_split_data_pickle()

    # For Label Spreading, some labels should be unlabeled (e.g., -1)
    # Here, we'll assume all training data is labeled. In practice, you'd have some unlabeled instances.
    # For demonstration, let's randomly mark some instances as unlabeled
    import numpy as np
    unlabeled_fraction = 0.1
    n_unlabeled = int(len(y_train) * unlabeled_fraction)
    np.random.seed(42)
    unlabeled_indices = np.random.choice(y_train.index, n_unlabeled, replace=False)
    y_train_unlabeled = y_train.copy()
    y_train_unlabeled.loc[unlabeled_indices] = -1  # Mark as unlabeled

    # Build and evaluate the Label Spreading model
    build_label_spreading_model(X_train, X_test, y_train_unlabeled, y_test)


if __name__ == "__main__":
    main()
