import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
def random_split_features(X, view_size_ratio=0.5):
    all_features = X.columns.tolist()
    random.shuffle(all_features)
    split_index = int(len(all_features) * view_size_ratio)
    view1_features = all_features[:split_index]
    view2_features = all_features[split_index:]
    return X[view1_features], X[view2_features], view1_features, view2_features


def pca_based_split(X, view_size_ratio=0.5):
    pca = PCA(n_components=min(X.shape[1], 2))
    pca.fit(X)
    feature_contributions = abs(pca.components_[0])
    feature_ranking = np.argsort(feature_contributions)[::-1]
    split_index = int(len(feature_ranking) * view_size_ratio)
    view1_features = X.columns[feature_ranking[:split_index]].tolist()
    view2_features = X.columns[feature_ranking[split_index:]].tolist()
    return X[view1_features], X[view2_features], view1_features, view2_features


def correlation_based_split(X, view_size_ratio=0.5):
    correlation_matrix = X.corr().abs()
    linkage_matrix = linkage(correlation_matrix, method='ward')
    clusters = fcluster(linkage_matrix, t=2, criterion='maxclust')
    view1_features = X.columns[clusters == 1].tolist()
    view2_features = X.columns[clusters == 2].tolist()
    # Adjust feature sizes based on view_size_ratio
    split_index = int(len(view1_features) * view_size_ratio)
    view1_features = view1_features[:split_index]
    view2_features = view2_features[:len(view2_features) - split_index]
    return X[view1_features], X[view2_features], view1_features, view2_features


def importance_based_split(X, y, view_size_ratio=0.5):
    # Align X and y lengths
    labeled_indices = y.index
    X = X.loc[labeled_indices]

    # Train Random Forest on labeled data
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    # Rank features by importance
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Split features based on importance
    split_index = int(len(feature_importances) * view_size_ratio)
    view1_features = feature_importances.index[:split_index].tolist()
    view2_features = feature_importances.index[split_index:].tolist()

    return X[view1_features], X[view2_features], view1_features, view2_features


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
        else:
            X_train_view1, X_train_view2, features_view1, features_view2 = method(
                X_train, view_size_ratio=view_size_ratio
            )

        X_test_view1 = X_test[features_view1]
        X_test_view2 = X_test[features_view2]

        clf1 = GradientBoostingClassifier(random_state=42)
        clf2 = GradientBoostingClassifier(random_state=42)

        labeled_mask = y_train != -1
        clf1.fit(X_train_view1[labeled_mask], y_train[labeled_mask])
        clf2.fit(X_train_view2[labeled_mask], y_train[labeled_mask])

        y_pred1 = clf1.predict(X_test_view1)
        y_pred2 = clf2.predict(X_test_view2)
        y_pred = (y_pred1 + y_pred2) >= 1  # Majority vote
        y_pred_proba = (clf1.predict_proba(X_test_view1)[:, 1] + clf2.predict_proba(X_test_view2)[:, 1]) / 2

        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)

        results.append({
            'method': method.__name__,
            'features_view1': features_view1,
            'features_view2': features_view2,
            'auc': auc,
            'accuracy': accuracy,
            'y_pred': y_pred.tolist(),
            'y_pred_proba': y_pred_proba.tolist()
        })

        print(f"Method: {method.__name__} | AUC: {auc:.4f} | Accuracy: {accuracy:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_method = method

    return best_method, results


def main():
    X_train, X_test, y_train, y_test, scaler = load_split_data_pickle()

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
    main()
