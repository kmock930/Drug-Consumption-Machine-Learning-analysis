# evaluate_models_with_varying_labeled_fractions.py

import pandas as pd
import numpy as np
import os
import pickle
from joblib import dump, load as joblib_load
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Function Definitions
# ------------------------------

def load_split_data_pickle(directory='split_data_pickle'):
    """
    Loads the split and processed data saved as pickle files.

    Returns:
    - X_train_full (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Testing features.
    - y_train_full (pd.Series): Training labels.
    - y_test (pd.Series): Testing labels.
    """
    filenames = ['X_train.pkl', 'X_test.pkl', 'y_train.pkl', 'y_test.pkl']
    loaded_data = {}

    for filename in filenames:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                loaded_data[filename] = pickle.load(file)
            print(f"Successfully loaded '{filename}' from '{directory}/'")
        else:
            raise FileNotFoundError(f"'{filename}' not found in directory '{directory}/'.")

    X_train_full = loaded_data['X_train.pkl']
    X_test = loaded_data['X_test.pkl']
    y_train_full = loaded_data['y_train.pkl']
    y_test = loaded_data['y_test.pkl']

    return X_train_full, X_test, y_train_full, y_test

def split_labeled_unlabeled(y_train, labeled_fraction=0.1, random_state=42):
    """
    Splits the training labels into labeled and unlabeled subsets, ensuring at least one labeled instance per class.

    Returns:
    - y_train_split (pd.Series): Training labels with unlabeled instances marked as -1.
    """
    np.random.seed(random_state)
    y_train_split = y_train.copy()
    n_samples = len(y_train)
    n_labeled = int(n_samples * labeled_fraction)

    # Ensure at least one instance per class is labeled
    labeled_indices = []

    for class_label in y_train.unique():
        class_indices = y_train[y_train == class_label].index
        if len(class_indices) == 0:
            continue
        labeled_index = np.random.choice(class_indices, size=1, replace=False)
        labeled_indices.extend(labeled_index)

    remaining_indices = y_train.index.difference(labeled_indices)
    n_additional = n_labeled - len(labeled_indices)
    if n_additional > 0 and len(remaining_indices) >= n_additional:
        additional_indices = np.random.choice(remaining_indices, size=n_additional, replace=False)
        labeled_indices.extend(additional_indices)
    elif n_additional > 0:
        # If not enough remaining indices, include all
        labeled_indices.extend(remaining_indices)

    y_train_split[:] = -1  # Mark all as unlabeled
    y_train_split.loc[labeled_indices] = y_train.loc[labeled_indices]  # Assign labels to selected indices

    print(f"Labeled {len(labeled_indices)} out of {n_samples} instances ({labeled_fraction*100}%).")
    return y_train_split

def train_and_evaluate_models(X_train_full, y_train_full, X_test, y_test, labeled_fractions, random_state=42):
    """
    Trains and evaluates models for different labeled fractions.

    Parameters:
    - X_train_full (pd.DataFrame): Full training features.
    - y_train_full (pd.Series): Full training labels.
    - X_test (pd.DataFrame): Testing features.
    - y_test (pd.Series): Testing labels.
    - labeled_fractions (list): List of labeled fractions to evaluate.
    - random_state (int): Random state for reproducibility.

    Returns:
    - results_df (pd.DataFrame): DataFrame containing evaluation results.
    """
    results = []

    for fraction in labeled_fractions:
        print(f"\n=== Evaluating with {int(fraction*100)}% Labeled Data ===")
        y_train_split = split_labeled_unlabeled(y_train_full, labeled_fraction=fraction, random_state=random_state)

        # Prepare datasets
        X_train_labeled = X_train_full[y_train_split != -1]
        y_train_labeled = y_train_split[y_train_split != -1]
        X_train_unlabeled = X_train_full[y_train_split == -1]
        y_train_unlabeled = y_train_split[y_train_split == -1]

        # Train Supervised Gradient Boosting
        print("\nTraining Supervised Gradient Boosting Model...")
        gb_model = GradientBoostingClassifier(random_state=random_state)
        gb_model.fit(X_train_labeled, y_train_labeled)
        metrics, y_pred, _ = evaluate_model(gb_model, X_test, y_test, model_name='Supervised Gradient Boosting')
        metrics['Model'] = 'Supervised Gradient Boosting'
        metrics['Labeled Fraction'] = fraction
        results.append(metrics)
        plot_confusion_matrix(y_test, y_pred, f'Gradient_Boosting_{int(fraction*100)}%_Labeled')

        # Train Self-Training Model
        print("\nTraining Self-Training Model...")
        base_estimator = GradientBoostingClassifier(random_state=random_state)
        self_training_model = SelfTrainingClassifier(base_estimator, verbose=False)
        self_training_model.fit(X_train_full, y_train_split)
        metrics, y_pred, _ = evaluate_model(self_training_model, X_test, y_test, model_name='Self-Training')
        metrics['Model'] = 'Self-Training'
        metrics['Labeled Fraction'] = fraction
        results.append(metrics)
        plot_confusion_matrix(y_test, y_pred, f'Self_Training_{int(fraction*100)}%_Labeled')

        # Train Label Spreading Model
        print("\nTraining Label Spreading Model...")
        label_spreading_model = LabelSpreading()
        label_spreading_model.fit(X_train_full, y_train_split)
        metrics, y_pred, _ = evaluate_model(label_spreading_model, X_test, y_test, model_name='Label Spreading')
        metrics['Model'] = 'Label Spreading'
        metrics['Labeled Fraction'] = fraction
        results.append(metrics)
        plot_confusion_matrix(y_test, y_pred, f'Label_Spreading_{int(fraction*100)}%_Labeled')

    results_df = pd.DataFrame(results)
    return results_df

def evaluate_model(model, X_test, y_test, average_method='weighted', model_name=''):
    """
    Evaluates the model and returns metrics.

    Returns:
    - metrics (dict): Evaluation metrics.
    - y_pred (np.array): Predicted labels.
    - y_pred_proba (np.array): Predicted probabilities.
    """
    y_pred = model.predict(X_test)
    num_classes = len(np.unique(y_test))
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
        if np.isnan(y_pred_proba).any():
            y_pred_proba = np.nan_to_num(y_pred_proba)
        if num_classes == 2:
            y_pred_proba = y_pred_proba[:, 1]
            average_param = 'binary'
            auc_score = roc_auc_score(y_test, y_pred_proba)
        else:
            average_param = average_method
            auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    else:
        y_pred_proba = None
        auc_score = None
        average_param = 'weighted' if num_classes > 2 else 'binary'

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=average_param, zero_division=0)
    recall = recall_score(y_test, y_pred, average=average_param, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average_param, zero_division=0)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_score
    }

    print(f"\nEvaluation Metrics for {model_name}:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: N/A")

    return metrics, y_pred, y_pred_proba

def plot_confusion_matrix(y_test, y_pred, model_name, directory='results'):
    """
    Plots and saves the confusion matrix.
    """
    os.makedirs(directory, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plot_path = os.path.join(directory, f'Confusion_Matrix_{model_name}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_metrics_over_fractions(results_df, directory='results'):
    """
    Plots evaluation metrics over labeled fractions.
    """
    os.makedirs(directory, exist_ok=True)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=results_df,
            x='Labeled Fraction',
            y=metric,
            hue='Model',
            marker='o'
        )
        plt.title(f'{metric} vs Labeled Fraction', fontsize=16)
        plt.xlabel('Labeled Fraction', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Model', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(directory, f'{metric}_vs_Labeled_Fraction.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Plot saved to '{plot_path}'.")

def save_results(results_df, filename='evaluation_results.csv', directory='results'):
    """
    Saves the evaluation results to a CSV file.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing the results.
    - filename (str): Name of the CSV file.
    - directory (str): Directory to save the results.
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    results_df.to_csv(file_path, index=False)
    print(f"Results successfully saved to '{file_path}'.")

# ------------------------------
# Main Evaluation Workflow
# ------------------------------

def main_evaluation():
    # Load data
    X_train_full, X_test, y_train_full, y_test = load_split_data_pickle()

    # Define labeled fractions
    labeled_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Train and evaluate models
    results_df = train_and_evaluate_models(
        X_train_full, y_train_full, X_test, y_test, labeled_fractions, random_state=42
    )

    # Save results
    save_results(results_df, filename='evaluation_results.csv', directory='results')

    # Plot metrics over fractions
    plot_metrics_over_fractions(results_df, directory='results')

    print("\n=== Evaluation Completed ===")

if __name__ == "__main__":
    main_evaluation()
