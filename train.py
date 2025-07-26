"""
Training script for California Housing dataset using Linear Regression
"""
import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import os

def load_data():
    """Load and split the California Housing dataset"""
    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train Linear Regression model"""
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"Model coefficients shape: {model.coef_.shape}")
    print(f"Model intercept: {model.intercept_:.4f}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    
    return r2, mse

def save_model_and_data(model, X_test, y_test):
    """Save the trained model and test data"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the trained model
    model_path = 'models/linear_regression_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save test data for prediction verification
    test_data_path = 'models/test_data.joblib'
    joblib.dump({'X_test': X_test, 'y_test': y_test}, test_data_path)
    print(f"Test data saved to {test_data_path}")

def main():
    """Main training pipeline"""
    print("Starting training pipeline...")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    r2, mse = evaluate_model(model, X_test, y_test)
    
    # Save model and test data
    save_model_and_data(model, X_test, y_test)
    
    print("Training pipeline completed successfully!")
    print(f"Final R² Score: {r2:.4f}")

if __name__ == "__main__":
    main()