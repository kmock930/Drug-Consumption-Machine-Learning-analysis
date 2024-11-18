import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier;
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import random
import numpy as np
import json
from scipy.cluster.hierarchy import linkage, fcluster
import joblib
from scipy import stats
import sys;
# importing custom modules
sys.path.append("../Assignment 1 - Model Comparison/");
sys.path.append("../Project - Semi Supervised Learning");
from performance import evalAccuracy, evalPrecision, evalRecall, evalF1Score, printConfMtx, plotROC;
import constants;

# Load Split Data
def load_split_data_pickle(directory='split_data_pickle'):
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
    return (
        loaded_data['X_train.pkl'],
        loaded_data['X_test.pkl'],
        loaded_data['y_train.pkl'],
        loaded_data['y_test.pkl'],
        loaded_data['scaler.pkl'],
    )


# Feature Splitting Methods
def random_split_features(X, y, view_size_ratio=constants.CO_TRAINING_FEATURE_SPLIT):
    """
    Perform a random split on features for co-training.

    Parameters:
    X (numpy.ndarray): The input feature matrix.
    y (numpy.ndarray): The labels.
    view_size_ratio (float): The ratio of features to be included in the first view.

    Returns:
    X_view1 (numpy.ndarray): The first view of the feature matrix.
    X_view2 (numpy.ndarray): The second view of the feature matrix.
    features_view1 (list): The indices of features in the first view.
    features_view2 (list): The indices of features in the second view.
    """
    num_features = X.shape[1]
    num_features_view1 = int(num_features * view_size_ratio)
    all_features = np.arange(num_features)
    np.random.shuffle(all_features)
    features_view1 = all_features[:num_features_view1]
    features_view2 = all_features[num_features_view1:]

    X_view1 = X[:, features_view1]
    X_view2 = X[:, features_view2]

    print("Number of features in view 1:", len(features_view1))
    print("Number of features in view 2:", len(features_view2))

    return X_view1, X_view2, features_view1, features_view2


def pca_based_split(X, y, n_components=2):
    """
    Perform PCA-based split on features.
    
    Parameters:
    X (numpy.ndarray): The input data with features to be split.
    n_components (int): Number of principal components to use for splitting.
    
    Returns:
    X_view1 (numpy.ndarray): The first view of the data after PCA split.
    X_view2 (numpy.ndarray): The second view of the data after PCA split.
    features_view1 (numpy.ndarray): Indices of features in the first view.
    features_view2 (numpy.ndarray): Indices of features in the second view.
    """
    pca = PCA(n_components=n_components)
    pca.fit(X, y)
    
    # Get the principal components
    components = pca.components_

    # Split the features based on the principal components
    features_view1 = np.argsort(components[0])[:X.shape[1] // 2]
    features_view2 = np.argsort(components[0])[X.shape[1] // 2:]
    
    X_view1 = X[:, features_view1]
    X_view2 = X[:, features_view2]

    print("Number of features in view 1:", len(features_view1))
    print("Number of features in view 2:", len(features_view2))
    
    return X_view1, X_view2, features_view1, features_view2


def correlation_based_split(X, y, view_size_ratio=0.5):
    """
    Perform correlation-based split on features.
    
    Parameters:
    X (numpy.ndarray): Feature matrix.
    y (numpy.ndarray): Labels.
    view_size_ratio (float): Ratio of features to be included in the first view.
    
    Returns:
    X_view1 (numpy.ndarray): Feature matrix for the first view.
    X_view2 (numpy.ndarray): Feature matrix for the second view.
    features_view1 (numpy.ndarray): Indices of features in the first view.
    features_view2 (numpy.ndarray): Indices of features in the second view.
    """
    # Calculate the correlation matrix
    corr_matrix = np.corrcoef(X, rowvar=False)
    
    # Get the absolute values of the correlation matrix
    abs_corr_matrix = np.abs(corr_matrix)
    
    # Sum the absolute correlations for each feature
    feature_corr_sums = np.sum(abs_corr_matrix, axis=0)
    
    # Get the number of features to include in the first view
    num_features_view1 = int(len(feature_corr_sums) * view_size_ratio)
    
    # Get the indices of the features with the highest correlation sums
    features_view1 = np.argsort(feature_corr_sums)[-num_features_view1:]
    
    # Get the indices of the remaining features for the second view
    features_view2 = np.setdiff1d(np.arange(len(feature_corr_sums)), features_view1)
    
    # Split the feature matrix into two views
    X_view1 = X[:, features_view1]
    X_view2 = X[:, features_view2]

    print("Features in View 1: ", features_view1)
    print("Features in View 2: ", features_view2)
    
    return X_view1, X_view2, features_view1, features_view2


def importance_based_split(X, y, view_size_ratio=0.5):
    """
    Perform importance-based split on features.
    
    Parameters:
    X (numpy.ndarray): Feature matrix.
    y (numpy.ndarray): Labels.
    view_size_ratio (float): Ratio of features to be included in the first view.
    
    Returns:
    X_view1 (numpy.ndarray): Feature matrix for the first view.
    X_view2 (numpy.ndarray): Feature matrix for the second view.
    features_view1 (numpy.ndarray): Indices of features in the first view.
    features_view2 (numpy.ndarray): Indices of features in the second view.
    """
    # Train a Gradient Boosting Classifier to get feature importances
    gbc = GradientBoostingClassifier(random_state=42)
    gbc.fit(X, y)
    
    # Get feature importances
    importances = gbc.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Split features based on the view size ratio
    split_index = int(len(indices) * view_size_ratio)
    features_view1 = indices[:split_index]
    features_view2 = indices[split_index:]
    
    # Split the dataset into two views
    X_view1 = X[:, features_view1]
    X_view2 = X[:, features_view2]

    print("Features in View 1: ", features_view1)
    print("Features in View 2: ", features_view2)
    
    return X_view1, X_view2, features_view1, features_view2


# Plotting Utilities
def plot_confusion_matrix(y_true, y_pred, method_name):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix ({method_name})')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{method_name}.png')
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, auc, method_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({method_name})')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'roc_curve_{method_name}.png')
    plt.show()


# Co-Training Framework
def evaluate_feature_split_methods(methods, X_train, X_test, y_train, y_test, view_size_ratio=0.5):
    results = []
    best_auc = -1
    best_method = None

    for method in methods:
        print(f"\n=== Evaluating Method: {method.__name__} ===")

        if method == importance_based_split:
            # Filter out unlabeled data for both X_train and y_train
            labeled_mask = y_train != -1
            X_train_view1, X_train_view2, features_view1, features_view2 = method(
                X_train[labeled_mask], y_train[labeled_mask], view_size_ratio=view_size_ratio
            )
        elif method == pca_based_split:
            X_train_view1, X_train_view2, features_view1, features_view2 = method(
                X_train, y_train # no need to provide view ratio
            )
        else:
            X_train_view1, X_train_view2, features_view1, features_view2 = method(
                X_train, y_train, view_size_ratio=view_size_ratio
            )

        X_test_view1 = X_test[:, features_view1]
        X_test_view2 = X_test[:, features_view2]

        clf1 = GradientBoostingClassifier(random_state=42)
        clf2 = GradientBoostingClassifier(random_state=42)

        labeled_mask = y_train != -1
        clf1.fit(X_train_view1[labeled_mask], y_train[labeled_mask])
        clf2.fit(X_train_view2[labeled_mask], y_train[labeled_mask])

        y_pred1 = clf1.predict(X_test_view1)
        y_pred2 = clf2.predict(X_test_view2)
        # Deduce the prediction labels by Majority Vote
        assert(len(y_pred1) == len(y_pred2));
        y_pred = [];
        for i in range(len(y_pred1)):
            y_pred.append(
                1 if y_pred1[i] + y_pred2[i] >= 1 else 0
            );
        #y_pred = 1 if (y_pred1 + y_pred2) >= 1 else 0 # Majority vote
        y_pred_proba = (clf1.predict_proba(X_test_view1)[:, 1] + clf2.predict_proba(X_test_view2)[:, 1]) / 2

        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)

        results.append({
            'method': method.__name__,
            'features_view1': features_view1.tolist(),
            'features_view2': features_view2.tolist(),
            'auc': auc,
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba.tolist()
        })

        print(f"Method: {method.__name__} | AUC: {auc:.4f} | Accuracy: {accuracy:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_method = method

    return best_method, results


def main():
    X_train = joblib.load(
        filename="X_train.pkl"
    );
    X_test = joblib.load(
        filename="X_test.pkl"
    );
    y_train = joblib.load(
        filename="y_train.pkl"
    );
    y_test = joblib.load(
        filename="y_test.pkl"
    )

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    unlabeled_fraction = 0.1
    n_unlabeled = int(len(y_train) * unlabeled_fraction)
    np.random.seed(42)
    unlabeled_indices = np.random.choice(y_train.index, n_unlabeled, replace=False)
    y_train.loc[unlabeled_indices] = -1  # Mark as unlabeled

    methods = [random_split_features, pca_based_split, correlation_based_split, importance_based_split]

    view_size_ratios = [0.3, 0.5, 0.7]  # Try different feature size ratios
    all_results = []

    for ratio in view_size_ratios:
        print(f"\n=== Evaluating with View Size Ratio: {ratio} ===")
        best_method, ratio_results = evaluate_feature_split_methods(
            methods, X_train_scaled, X_test_scaled, y_train, y_test, view_size_ratio=ratio
        )
        all_results.extend(ratio_results)

    with open('feature_split_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    best_result = max(all_results, key=lambda x: x['auc'])
    plot_confusion_matrix(y_test, best_result['y_pred'], best_result['method'])
    plot_roc_curve(y_test, best_result['y_pred_proba'], best_result['auc'], best_result['method'])

    print("\nBest Method:", best_result['method'])
    print("Best Result:", best_result)


if __name__ == "__main__":
    pass
    #main()
