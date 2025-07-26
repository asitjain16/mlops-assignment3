"""
Manual quantization script for the trained Linear Regression model
Converts float32 parameters to 8-bit unsigned integers
"""
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
import os

class SimpleLinearModel(nn.Module):
    """Simple PyTorch linear model"""
    def __init__(self, input_size):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return self.linear(x).squeeze()

def load_sklearn_model():
    """Load the trained scikit-learn model"""
    model_path = 'models/linear_regression_model.joblib'
    test_data_path = 'models/test_data.joblib'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print("Loading trained scikit-learn model...")
    model = joblib.load(model_path)
    test_data = joblib.load(test_data_path)
    
    return model, test_data['X_test'], test_data['y_test']

def extract_parameters(model):
    """Extract parameters from scikit-learn model"""
    print("Extracting model parameters...")
    
    coef = model.coef_.astype(np.float32)
    intercept = model.intercept_.astype(np.float32)  # Keep as scalar, not array
    
    print(f"Coefficients shape: {coef.shape}")
    print(f"Coefficients range: [{coef.min():.6f}, {coef.max():.6f}]")
    print(f"Intercept: {intercept:.6f}")
    
    return coef, intercept

def manual_quantization(params, bits=8):
    """
    Manual quantization to unsigned 8-bit integers
    Formula: quantized = round((param - min_val) / (max_val - min_val) * (2^bits - 1))
    """
    print(f"Performing manual quantization to {bits}-bit unsigned integers...")
    
    quantized_params = {}
    scale_zero_point = {}
    
    for name, param in params.items():
        # Find min and max values
        min_val = float(param.min())
        max_val = float(param.max())
        
        # Handle case where min_val == max_val (constant value)
        if abs(max_val - min_val) < 1e-10:
            # For constant values, store the original value and skip quantization
            quantized_params[name] = param.astype(np.float32)  # Keep original
            scale_zero_point[name] = {
                'scale': 1.0,
                'zero_point': 0,
                'min_val': min_val,
                'max_val': max_val,
                'is_constant': True
            }
            print(f"{name}:")
            print(f"  Constant value: {min_val:.6f} (skipping quantization)")
            continue
        
        # Calculate scale and zero point
        scale = (max_val - min_val) / (2**bits - 1)
        zero_point = 0  # For unsigned quantization
        
        # Quantize: q = round((r - min_val) / scale) + zero_point
        quantized = np.round((param - min_val) / scale).astype(np.uint8)
        
        quantized_params[name] = quantized
        scale_zero_point[name] = {
            'scale': scale,
            'zero_point': zero_point,
            'min_val': min_val,
            'max_val': max_val,
            'is_constant': False
        }
        
        print(f"{name}:")
        print(f"  Original range: [{min_val:.6f}, {max_val:.6f}]")
        print(f"  Scale: {scale:.8f}")
        print(f"  Quantized range: [{quantized.min()}, {quantized.max()}]")
    
    return quantized_params, scale_zero_point

def dequantize_parameters(quantized_params, scale_zero_point):
    """
    Dequantize parameters back to float32
    Formula: dequantized = scale * (quantized - zero_point) + min_val
    """
    print("Dequantizing parameters...")
    
    dequantized_params = {}
    
    for name, quantized in quantized_params.items():
        info = scale_zero_point[name]
        
        if info.get('is_constant', False):
            # For constant values, just use the original value
            dequantized_params[name] = quantized
            print(f"{name} dequantized (constant): {quantized.flatten()[0]:.6f}")
        else:
            scale = info['scale']
            zero_point = info['zero_point']
            min_val = info['min_val']
            
            # Dequantize: r = scale * (q - zero_point) + min_val
            dequantized = scale * (quantized.astype(np.float32) - zero_point) + min_val
            dequantized_params[name] = dequantized
            
            print(f"{name} dequantized range: [{dequantized.min():.6f}, {dequantized.max():.6f}]")
    
    return dequantized_params

def create_pytorch_model(coef, intercept, input_size):
    """Create PyTorch model and set weights manually"""
    print("Creating PyTorch model...")
    
    model = SimpleLinearModel(input_size)
    
    # Set weights manually
    with torch.no_grad():
        model.linear.weight.data = torch.from_numpy(coef.reshape(1, -1))
        if intercept.ndim == 0:
            model.linear.bias.data = torch.tensor([intercept])
        else:
            model.linear.bias.data = torch.from_numpy(intercept.flatten())
    
    print(f"  Weight shape: {model.linear.weight.data.shape}")
    print(f"  Bias shape: {model.linear.bias.data.shape}")
    print(f"  Weight range: [{model.linear.weight.data.min():.6f}, {model.linear.weight.data.max():.6f}]")
    print(f"  Bias value: {model.linear.bias.data[0]:.6f}")
    
    return model

def evaluate_pytorch_model(model, X_test, y_test):
    """Evaluate PyTorch model"""
    print("Evaluating PyTorch model...")
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test.astype(np.float32))
        y_pred = model(X_tensor).numpy()
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"PyTorch model R² Score: {r2:.6f}")
    print(f"PyTorch model MSE: {mse:.6f}")
    
    return r2, mse

def calculate_model_sizes():
    """Calculate file sizes of saved models"""
    unquant_path = 'models/unquant_params.joblib'
    quant_path = 'models/quant_params.joblib'
    simple_quant_path = 'models/simple_quant_params.joblib'
    
    unquant_size = os.path.getsize(unquant_path) / 1024  # KB
    quant_size = os.path.getsize(quant_path) / 1024  # KB
    simple_quant_size = os.path.getsize(simple_quant_path) / 1024  # KB
    
    print(f"\nModel size comparison:")
    print(f"Unquantized model: {unquant_size:.2f} KB")
    print(f"Quantized model (with metadata): {quant_size:.2f} KB")
    print(f"Simple quantized model: {simple_quant_size:.2f} KB")
    print(f"Size reduction (simple): {((unquant_size - simple_quant_size) / unquant_size * 100):.1f}%")
    
    # Use simple quantized size for the comparison table
    return unquant_size, simple_quant_size

def main():
    """Main quantization pipeline"""
    print("Starting quantization pipeline...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load sklearn model
    sklearn_model, X_test, y_test = load_sklearn_model()
    
    # Extract parameters
    coef, intercept = extract_parameters(sklearn_model)
    
    # Store unquantized parameters
    unquant_params = {
        'coef': coef,
        'intercept': np.array([intercept])  # Convert to array for consistent handling
    }
    
    unquant_path = 'models/unquant_params.joblib'
    joblib.dump(unquant_params, unquant_path)
    print(f"Unquantized parameters saved to {unquant_path}")
    
    # Perform manual quantization
    quantized_params, scale_zero_point = manual_quantization(unquant_params)
    
    # Save quantized parameters
    quant_data = {
        'quantized_params': quantized_params,
        'scale_zero_point': scale_zero_point
    }
    
    quant_path = 'models/quant_params.joblib'
    joblib.dump(quant_data, quant_path)
    print(f"Quantized parameters saved to {quant_path}")
    
    # Also save just the quantized parameters for size comparison
    simple_quant_data = {
        'coef_quantized': quantized_params['coef'],
        'intercept': quantized_params['intercept']  # This is already float32 for constant
    }
    simple_quant_path = 'models/simple_quant_params.joblib'
    joblib.dump(simple_quant_data, simple_quant_path)
    print(f"Simple quantized parameters saved to {simple_quant_path}")
    
    # Dequantize for inference
    dequantized_params = dequantize_parameters(quantized_params, scale_zero_point)
    
    # Create PyTorch models
    input_size = coef.shape[0]
    
    # Original model
    original_model = create_pytorch_model(coef, np.array([intercept]), input_size)
    original_r2, original_mse = evaluate_pytorch_model(original_model, X_test, y_test)
    
    # Debug: Check dequantized values
    print(f"\nDebugging dequantized values:")
    print(f"Original coef range: [{coef.min():.6f}, {coef.max():.6f}]")
    print(f"Dequantized coef range: [{dequantized_params['coef'].min():.6f}, {dequantized_params['coef'].max():.6f}]")
    print(f"Original intercept: {intercept:.6f}")
    print(f"Dequantized intercept: {dequantized_params['intercept'].flatten()[0]:.6f}")
    
    # Check difference
    coef_diff = np.abs(coef - dequantized_params['coef']).max()
    intercept_diff = np.abs(intercept - dequantized_params['intercept'].flatten()[0])
    print(f"Max coefficient difference: {coef_diff:.8f}")
    print(f"Intercept difference: {intercept_diff:.8f}")
    
    # Quantized model (using dequantized weights)
    quantized_model = create_pytorch_model(
        dequantized_params['coef'], 
        dequantized_params['intercept'], 
        input_size
    )
    quantized_r2, quantized_mse = evaluate_pytorch_model(quantized_model, X_test, y_test)
    
    # Calculate model sizes
    unquant_size, quant_size = calculate_model_sizes()
    
    # Print comparison table
    print("\n" + "="*60)
    print("QUANTIZATION RESULTS COMPARISON")
    print("="*60)
    print(f"{'Metric':<20} {'Original Model':<15} {'Quantized Model':<15}")
    print("-"*60)
    print(f"{'R² Score':<20} {original_r2:<15.6f} {quantized_r2:<15.6f}")
    print(f"{'Model Size (KB)':<20} {unquant_size:<15.2f} {quant_size:<15.2f}")
    print("="*60)
    
    print("\nQuantization pipeline completed successfully!")

if __name__ == "__main__":
    main()