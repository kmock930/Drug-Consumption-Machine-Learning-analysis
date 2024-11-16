# model_building.py

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from joblib import dump, load as joblib_load


# ------------------------------
# Function Definitions
# ------------------------------

def load_split_data_pickle(directory='split_data_pickle'):
    """
    Loads the split and processed data saved as pickle files.

    Parameters:
    - directory (str): The directory where the pickle files are stored.

    Returns:
    - X_train (pd.DataFrame): Scaled training features.
    - X_test (pd.DataFrame): Scaled testing features.
    - y_train (pd.Series): Training labels.
    - y_test (pd.Series): Testing labels.
    - scaler (StandardScaler): Fitted scaler object.
    """
    # Define the filenames to load
    filenames = ['X_train.pkl', 'X_test.pkl', 'y_train.pkl', 'y_test.pkl', 'scaler.pkl']

    # Initialize a dictionary to hold the loaded objects
    loaded_data = {}

    # Iterate over each file and load its content
    for filename in filenames:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                loaded_data[filename] = pickle.load(file)
            print(f"Successfully loaded '{filename}' from '{directory}/'")
        else:
            raise FileNotFoundError(
                f"'{filename}' not found in directory '{directory}/'. Please ensure the file exists.")

    # Extract the loaded objects
    X_train = loaded_data['X_train.pkl']
    X_test = loaded_data['X_test.pkl']
    y_train = loaded_data['y_train.pkl']
    y_test = loaded_data['y_test.pkl']
    scaler = loaded_data['scaler.pkl']

    return X_train, X_test, y_train, y_test, scaler


def save_model_pickle(model, directory='models'):
    """
    Saves the trained model as a pickle file.

    Parameters:
    - model: Trained machine learning model.
    - directory (str): Directory to save the model.
    """
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, 'gradient_boosting_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model successfully saved to '{model_path}' using pickle.")


def save_model_joblib(model, directory='models'):
    """
    Saves the trained model as a joblib file.

    Parameters:
    - model: Trained machine learning model.
    - directory (str): Directory to save the model.
    """
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, 'gradient_boosting_model.joblib')
    dump(model, model_path)
    print(f"Model successfully saved to '{model_path}' using joblib.")


def load_model_pickle(directory='models'):
    """
    Loads the trained model from a pickle file.

    Parameters:
    - directory (str): Directory where the model is saved.

    Returns:
    - model: Loaded machine learning model.
    """
    model_path = os.path.join(directory, 'gradient_boosting_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Successfully loaded model from '{model_path}' using pickle.")
        return model
    else:
        raise FileNotFoundError(f"Model file not found in directory '{directory}/'.")


def load_model_joblib_func(directory='models'):
    """
    Loads the trained model from a joblib file.

    Parameters:
    - directory (str): Directory where the model is saved.

    Returns:
    - model: Loaded machine learning model.
    """
    model_path = os.path.join(directory, 'gradient_boosting_model.joblib')
    if os.path.exists(model_path):
        model = joblib_load(model_path)
        print(f"Successfully loaded model from '{model_path}' using joblib.")
        return model
    else:
        raise FileNotFoundError(f"Model file not found in directory '{directory}/'.")


def build_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Builds a Gradient Boosting model using Grid Search for hyperparameter tuning,
    evaluates it on the test set, and visualizes the results including AUC-ROC.
    Also saves the trained model using both pickle and joblib.
    """
    from sklearn.model_selection import GridSearchCV

    # Define the parameter grid for Grid Search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0],
        'min_samples_split': [2, 5, 10]
    }

    # Initialize the Gradient Boosting Classifier
    gb_clf = GradientBoostingClassifier(random_state=42)

    # Initialize Grid Search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=gb_clf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    # Perform Grid Search on the training data
    print("\nStarting Grid Search for Hyperparameter Tuning...")
    grid_search.fit(X_train, y_train)

    # Display the best parameters and best score
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    # Retrieve the best model from Grid Search
    best_gb_clf = grid_search.best_estimator_

    # Save the trained model using both pickle and joblib
    save_model_pickle(best_gb_clf, directory='models')
    save_model_joblib(best_gb_clf, directory='models')

    # Make predictions on the test set
    y_pred = best_gb_clf.predict(X_test)
    y_pred_proba = best_gb_clf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # Calculate accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Set Accuracy: {test_accuracy:.4f}")

    # Calculate AUC-ROC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC Score: {auc:.4f}")

    # Detailed classification report
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
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def load_data():
    """
    Loads the preserved split data.

    Returns:
    - X_train (pd.DataFrame): Scaled training features.
    - X_test (pd.DataFrame): Scaled testing features.
    - y_train (pd.Series): Training labels.
    - y_test (pd.Series): Testing labels.
    - scaler (StandardScaler): Fitted scaler object.
    """
    # Using pickle
    X_train, X_test, y_train, y_test, scaler = load_split_data_pickle(directory='split_data_pickle')

    # OR using joblib
    # X_train, X_test, y_train, y_test, scaler = load_split_data_joblib(directory='split_data_joblib')

    return X_train, X_test, y_train, y_test, scaler


def model_workflow():
    """
    Executes the full model workflow:
    1. Loads the preserved data.
    2. Builds and evaluates the Gradient Boosting model, including AUC-ROC.
    """
    # Load the preserved data
    X_train, X_test, y_train, y_test, scaler = load_data()

    # Build and evaluate the model
    build_and_evaluate_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    model_workflow()
