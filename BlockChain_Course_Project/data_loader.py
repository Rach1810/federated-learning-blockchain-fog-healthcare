import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

def load_dataset():
    base_path = "UCI HAR Dataset"
    X_path = os.path.join(base_path, "train", "X_train.txt")
    y_path = os.path.join(base_path, "train", "y_train.txt")
    X_test_path = os.path.join(base_path, "test", "X_test.txt")
    y_test_path = os.path.join(base_path, "test", "y_test.txt")

    # Load data with error handling
    try:
        X = np.loadtxt(X_path)
        y = np.loadtxt(y_path)
        X_test = np.loadtxt(X_test_path)
        y_test = np.loadtxt(y_test_path)
    except Exception as e:
        raise FileNotFoundError(f"Error loading dataset: {e}")

    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Encode labels and scale features
    scaler = StandardScaler()
    label_encoder = LabelEncoder()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    return X_train, y_train, X_test, y_test