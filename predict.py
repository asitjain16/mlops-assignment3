"""
Prediction script to verify the Docker container
Loads the saved model and runs it on test set
"""
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os

def load_model_and_data():
    """Load the trained model and test data"""
    model_path = 'models/linear_regression_model.joblib'
    test_data_path = 'models/test_data.joblib'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    
    print("Loading trained model...")
    model = joblib.load(model_path)
    
    print("Loading test data...")
    test_data = joblib.load(test_data_path)
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    return model, X_test, y_test

def run_predictions(model, X_test, y_test):
    """Run predictions and evaluate performance"""
    print("Running predictions on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Test set size: {len(y_test)} samples")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    
    # Show some sample predictions
    print("\nSample predictions:")
    for i in range(min(5, len(y_test))):
        print(f"  Actual: {y_test[i]:.3f}, Predicted: {y_pred[i]:.3f}")
    
    return r2, mse

def verify_model_parameters(model):
    """Verify model parameters for later quantization"""
    print("\nModel parameter verification:")
    print(f"Coefficients shape: {model.coef_.shape}")
    print(f"Coefficients range: [{model.coef_.min():.4f}, {model.coef_.max():.4f}]")
    print(f"Intercept: {model.intercept_:.4f}")
    
    return model.coef_, model.intercept_

def main():
    """Main prediction pipeline"""
    print("Starting prediction verification...")
    
    try:
        # Load model and data
        model, X_test, y_test = load_model_and_data()
        
        # Run predictions
        r2, mse = run_predictions(model, X_test, y_test)
        
        # Verify parameters
        coef, intercept = verify_model_parameters(model)
        
        print("\nContainer verification completed successfully!")
        print(f"Model is working correctly with R² = {r2:.4f}")
        
        # Return success code
        return 0
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)