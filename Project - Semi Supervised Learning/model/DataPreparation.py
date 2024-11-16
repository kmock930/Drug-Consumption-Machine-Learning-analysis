# data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ------------------------------
# Function Definitions
# ------------------------------

def transform_label(series):
    """
    Transforms categorical labels into binary classification.

    Parameters:
    - series (pd.Series): Series containing categorical labels.

    Returns:
    - pd.Series: Binary labels where 'CL0' and 'CL1' are mapped to 0, others to 1.
    """
    return series.apply(lambda x: 0 if x in ['CL0', 'CL1'] else 1)

def load_and_process_data(filename):
    """
    Loads and processes the dataset.

    Parameters:
    - filename (str): Path to the CSV dataset.

    Returns:
    - X (pd.DataFrame): Feature dataframe.
    - y_choc (pd.Series): Binary labels for 'Choc'.
    - y_mushrooms (pd.Series): Binary labels for 'Mushrooms'.
    """
    # Load data
    data = pd.read_csv(filename)

    # Define label columns
    choc_label = 'Choc'          # Column name for 'Choc' usage
    mushroom_label = 'Mushrooms' # Column name for 'Mushrooms' usage

    # Transform labels
    data['Choc_binary'] = transform_label(data[choc_label])
    data['Mushrooms_binary'] = transform_label(data[mushroom_label])

    # Define features and labels
    # Assuming that columns 1 to 12 are feature columns; adjust as needed
    X = data.iloc[:, 1:13].apply(pd.to_numeric, errors='coerce')

    # Handle potential missing values after conversion
    X = X.fillna(0)  # Simple strategy; consider more sophisticated methods if needed

    y_choc = data['Choc_binary']
    y_mushrooms = data['Mushrooms_binary']

    return X, y_choc, y_mushrooms

def split_data(X, y, test_size=0.33, random_state=42):
    """
    Splits the data into training and testing sets with stratification.

    Parameters:
    - X (pd.DataFrame): Feature dataframe.
    - y (pd.Series): Labels.
    - test_size (float): Proportion of the dataset to include in the test split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Testing features.
    - y_train (pd.Series): Training labels.
    - y_test (pd.Series): Testing labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def normalize_data(X_train, X_test):
    """
    Normalizes the feature data using StandardScaler.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Testing features.

    Returns:
    - X_train_scaled_df (pd.DataFrame): Scaled training features.
    - X_test_scaled_df (pd.DataFrame): Scaled testing features.
    - scaler (StandardScaler): Fitted scaler object.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit only on training data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame with original column names and indices
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled_df, X_test_scaled_df, scaler

def save_split_data_pickle(X_train, X_test, y_train, y_test, scaler, directory='split_data_pickle'):
    """
    Saves the split and processed data as pickle files.

    Parameters:
    - X_train (pd.DataFrame): Scaled training features.
    - X_test (pd.DataFrame): Scaled testing features.
    - y_train (pd.Series): Training labels.
    - y_test (pd.Series): Testing labels.
    - scaler (StandardScaler): Fitted scaler object.
    - directory (str): Directory to save the pickle files.
    """
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'X_train.pkl'), 'wb') as f:
        pickle.dump(X_train, f)
    with open(os.path.join(directory, 'X_test.pkl'), 'wb') as f:
        pickle.dump(X_test, f)
    with open(os.path.join(directory, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(directory, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    with open(os.path.join(directory, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Data successfully saved to '{directory}' directory using pickle.")

# ------------------------------
# Main Workflow
# ------------------------------

def prepare_and_save_data():
    """
    Executes the full data preparation workflow:
    1. Loads and processes the data.
    2. Splits the data into training and testing sets.
    3. Normalizes the data.
    4. Saves the split and processed data using pickle.
    """
    filename = 'drug_consumption_processed.csv'  # Replace with your actual file path

    # Step 1: Load and process data
    X, y_choc, y_mushrooms = load_and_process_data(filename)

    # Step 2: Choose the target label
    y = y_mushrooms  # Change to y_choc if you want to model 'Choc_binary'

    # Step 3: Split the data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.33, random_state=42)
    print(f"Data split into training and testing sets:")
    print(f" - Training set size: {X_train.shape[0]} samples")
    print(f" - Testing set size: {X_test.shape[0]} samples")

    # Step 4: Normalize the data
    X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
    print("Data normalization complete.")

    # Step 5: Save the split and processed data
    save_split_data_pickle(X_train_scaled, X_test_scaled, y_train, y_test, scaler, directory='split_data_pickle')
    print("Data preparation and preservation complete.")

if __name__ == "__main__":
    prepare_and_save_data()
