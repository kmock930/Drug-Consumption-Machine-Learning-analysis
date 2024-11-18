'''
@Author: Yingshi Chen
'''

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)
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
    labeled_indices = y.index
    X = X.loc[labeled_indices]

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    split_index = int(len(feature_importances) * view_size_ratio)
    view1_features = feature_importances.index[:split_index].tolist()
    view2_features = feature_importances.index[split_index:].tolist()

    return X[view1_features], X[view2_features], view1_features, view2_features

# Evaluation Function
def evaluate_feature_split_methods(methods, X_train, X_test, y_train, y_test, view_size_ratio=0.5):
    results = []
    for method in methods:
        print(f"\n=== Evaluating Method: {method.__name__} ===")

        if method == importance_based_split:
            labeled_mask = y_train != -1
            X_train_labeled = X_train[labeled_mask]
            y_train_labeled = y_train[labeled_mask]
            X_train_view1, X_train_view2, features_view1, features_view2 = method(
                X_train_labeled, y_train_labeled, view_size_ratio=view_size_ratio
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
        X_train_view1_labeled = X_train_view1.loc[labeled_mask]
        X_train_view2_labeled = X_train_view2.loc[labeled_mask]
        y_train_labeled = y_train[labeled_mask]

        clf1.fit(X_train_view1_labeled, y_train_labeled)
        clf2.fit(X_train_view2_labeled, y_train_labeled)

        y_pred1 = clf1.predict(X_test_view1)
        y_pred2 = clf2.predict(X_test_view2)
        y_pred = (y_pred1 + y_pred2) >= 1  # Majority vote
        y_pred_proba = (
            clf1.predict_proba(X_test_view1)[:, 1] + clf2.predict_proba(X_test_view2)[:, 1]
        ) / 2

        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            'method': method.__name__,
            'view_size_ratio': view_size_ratio,
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_pred_proba': y_pred_proba.tolist()
        })

        print(
            f"Method: {method.__name__} | AUC: {auc:.4f} | Accuracy: {accuracy:.4f} | "
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}"
        )

    return results

# Visualization of Combined ROC-AUC
def plot_combined_roc_auc(results_df, y_test, view_size_ratio):
    plt.figure(figsize=(10, 6))
    methods = results_df['method'].unique()

    for method in methods:
        method_data = results_df[
            (results_df['method'] == method) & (results_df['view_size_ratio'] == view_size_ratio)
        ]
        if not method_data.empty:
            best_result = method_data.iloc[0]
            y_pred_proba = np.array(best_result['y_pred_proba'])
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

            plt.plot(fpr, tpr, label=f'{method} (AUC = {best_result["auc"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.title(f'ROC Curves for All Methods at View Size Ratio {view_size_ratio}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'ROC_Comparison_Ratio_{view_size_ratio}.png')
    plt.show()

# Visualization of Metrics
def visualize_metrics(results_df):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    methods_list = results_df['method'].unique()

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for method in methods_list:
            method_data = results_df[results_df['method'] == method]
            plt.plot(
                method_data['view_size_ratio'],
                method_data[metric],
                marker='o',
                label=method
            )
        plt.title(f'{metric.capitalize()} vs. View Size Ratio for Different Methods')
        plt.xlabel('View Size Ratio')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{metric.capitalize()}_vs_View_Size_Ratio.png')
        plt.show()

def main():
    X_train, X_test, y_train, y_test, scaler = load_split_data_pickle()

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    unlabeled_fraction = 0.1
    n_unlabeled = int(len(y_train) * unlabeled_fraction)
    np.random.seed(42)
    unlabeled_indices = np.random.choice(y_train.index, n_unlabeled, replace=False)
    y_train.loc[unlabeled_indices] = -1

    methods = [random_split_features, pca_based_split, correlation_based_split, importance_based_split]
    view_size_ratios = [0.3, 0.5, 0.7]
    all_results = []

    for ratio in view_size_ratios:
        print(f"\n=== Evaluating with View Size Ratio: {ratio} ===")
        results = evaluate_feature_split_methods(methods, X_train_scaled, X_test_scaled, y_train, y_test, view_size_ratio=ratio)
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    # Plot combined ROC-AUC for each view size ratio
    for ratio in view_size_ratios:
        plot_combined_roc_auc(results_df, y_test, ratio)

    # Visualize metrics
    visualize_metrics(results_df)

    print("\nAnalysis complete. Plots have been generated.")

if __name__ == "__main__":
    main()
