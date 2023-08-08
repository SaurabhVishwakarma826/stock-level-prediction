# import the reqaured library
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


# Load data from a CSV file
def load_data(path):
    df = pd.read_csv(path)
    # Assume that data is allready cleaned and transformed that why I directly return the dataframe after loading
    return df


# Separate the dataset into features (X) and target (Y)
def input_output(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# Train the RandomForestRegressor algorithm and evaluate its performance
def train_algorithm(X, Y):
    accuracy = []
    K = 10  # Number of folds for cross-validation
    split = 0.75  # Train-test split ratio

    for fold in range(0, K):
        # Instantiate the algorithm
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Create training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=split, random_state=42)

        # Scale X data to help the algorithm converge
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on the test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

    # Print average MAE across all folds
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")


if __name__ == '__main__':
    # Input the path to the CSV file
    path = input("Enter the path of the CSV file with its name and extension: ")

    # Load data from the CSV file
    df = load_data(path)

    # Extract features (X) and target (Y) from the dataset
    x, y = input_output(df, 'estimated_stock_pct')

    # Train the algorithm and evaluate its performance
    train_algorithm(x, y)
