from sklearn.ensemble import AdaBoostClassifier;
from sklearn.model_selection import RandomizedSearchCV
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from joblib import dump
import numpy as np
import constants

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


def save_model(model, directory='models', filename='semi_boost_model.joblib'):
    """
    Saves the trained model using Joblib.
    """
    os.makedirs(directory, exist_ok=True)
    model_path = os.path.join(directory, filename)
    dump(model, model_path)
    print(f"Model saved to '{model_path}'")


def build_semi_boost_model(baseModel=None):
    """
    Builds and evaluates a SemiBoost-like model using AdaBoost with self-training.
    """
    # Initialize base estimator as a weak learner
    defaultLearner = DecisionTreeClassifier(
        max_depth=1, 
        random_state=constants.RANDOM_STATE
    );
    base_estimator = baseModel if baseModel != None else defaultLearner;

    # Initialize AdaBoost Classifier with the weak learner
    ada_clf = AdaBoostClassifier(
        base_estimator=base_estimator, 
        n_estimators=constants.N_ESTIMATORS, 
        random_state=constants.RANDOM_STATE
    );

    # Initialize Self-Training Classifier with AdaBoost as base
    self_training_clf = SelfTrainingClassifier(
        base_estimator=ada_clf, 
        threshold=constants.THRESHOLD, 
        max_iter=constants.MAX_ITER, 
        verbose=True
    );

    save_model(
        model=self_training_clf, 
        directory=constants.MODEL_DIR, 
        filename=constants.MODEL_SEMIBOOST_FILENAME
    );

    return self_training_clf;

def train(model, X_train: np.ndarray, y_train: np.ndarray):
    '''
    Train the model
    '''
    # Fit the model
    print("\nTraining SemiBoost-like Classifier...")
    model.fit(X_train, y_train)

    # Save the model
    save_model(model, directory='models', filename=constants.MODEL_SEMIBOOST_FILENAME)

    return model

def predict(X_test: np.ndarray, y_test: np.ndarray, model):
    '''
    Predict the model
    '''
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    return y_pred, y_pred_proba;

def evaluate(y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray):
    '''
    Evaluate the model
    '''
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"SemiBoost Test Accuracy: {accuracy:.4f}")
    print(f"SemiBoost AUC-ROC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('SemiBoost Confusion Matrix')
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SemiBoost ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
