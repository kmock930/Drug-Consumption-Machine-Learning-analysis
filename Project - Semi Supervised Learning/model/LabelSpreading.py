'''
@author: Yixing Chen
'''
import pandas as pd
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from joblib import dump
from sklearn.model_selection import RandomizedSearchCV;
from imblearn.pipeline import Pipeline;
import constants;
from sklearn.base import BaseEstimator, TransformerMixin

# Define a custom transformer to use the predictions from the first model as features for the second model
class ProbabilitiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        return self.model.predict_proba(X)

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


def build_label_spreading_model(origModel):
    """
    Builds and evaluates a Label Spreading model.
    """
    # Initialize Label Spreading model
    labelSpread = LabelSpreading(
        kernel=constants.KERNEL, 
        gamma=constants.GAMMA, 
        max_iter=constants.MAX_ITER
    )

    # Apply Parameter tuning to label spreading
    labelSpread = RandomizedSearchCV(
        estimator=labelSpread,
        param_distributions=constants.LABEL_SPREAD_PARAM_DIST,
        n_iter=100,
        random_state=constants.RANDOM_STATE
    )

    # Create the custom transformer with the tuned Gradient Boosting model
    prob_transformer = ProbabilitiesTransformer(origModel.best_estimator_)

    # Create a pipeline with the original model and Label Spreading
    pipeline = Pipeline([
        ("original model", prob_transformer),
        ('label_spread', labelSpread)
    ])
    
    # Save the model
    save_model(pipeline, directory=constants.MODEL_DIR, filename=constants.MODEL_LABELSPREAD_FILENAME)

    return pipeline;

def train(model: Pipeline | LabelSpreading | RandomizedSearchCV, X_train, y_train):
    """
    Trains the Label Spreading model.
    """
    model.fit(X_train, y_train)
    save_model(model, directory=constants.MODEL_DIR, filename=constants.MODEL_LABELSPREAD_FILENAME)
    return model;

def predict(model: Pipeline | LabelSpreading | RandomizedSearchCV, X_test):
    """
    Predicts using the Label Spreading model.
    """
    return model.predict(X_test);

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
