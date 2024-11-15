import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load data function
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Main function
def main():
    # Replace this with the path to your CSV file
    file_path = "fraudTest.csv"  # Update with the actual file path

    # Load data
    data = load_data(file_path)
    if data is None:
        return

    # Inspect data to identify columns
    print("Data columns:", data.columns)
    print(data.head())

    # Assuming the dataset has 'isFraud' as the target and transaction-related columns as features
    # Replace 'isFraud' and other column names with those in your dataset
    if 'isFraud' not in data.columns:
        print("Expected 'isFraud' column not found in the dataset.")
        return

    # Feature selection (assuming 'isFraud' is the target column)
    X = data.drop(['isFraud'], axis=1)  # Drop target column from features
    y = data['isFraud']  # Define the target

    # Fill missing values, if any
    X = X.fillna(X.mean())

    # Standardize features for consistent model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Run the main function
if __name__ == "__main__":
    main()
