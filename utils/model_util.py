import numpy as np

# Local application imports
from network import (
    LinearLayer)

def save_model_weights(model, filename):
    """Save model weights to a file."""
    weights_dict = {}
    
    # Filter for LinearLayers first
    linear_layers = [m for m in model.modules() if isinstance(m, LinearLayer)]
    
    # Save each LinearLayer's weights
    for i, layer in enumerate(linear_layers):
        if layer.weights is not None and layer.biases is not None:
            weights = layer.weights.storage.to_numpy()
            biases = layer.biases.storage.to_numpy()
            weights_dict[f"layer_{i}_weights"] = weights
            weights_dict[f"layer_{i}_biases"] = biases
    
    # Save to file
    np.savez(filename, **weights_dict)
    print(f"Model weights saved to {filename} with {len(linear_layers)} linear layers")

def load_model_weights(model, filename):
    """Load model weights from a file."""
    # Load the weights
    weights_dict = np.load(filename)
    
    # Find all LinearLayers in the model
    linear_layers = [m for m in model.modules() if isinstance(m, LinearLayer)]
    
    # Assign weights to each layer
    for i, layer in enumerate(linear_layers):
        if f"layer_{i}_weights" in weights_dict and f"layer_{i}_biases" in weights_dict:
            # Copy weights and biases to the device
            layer.weights.storage.copy_from_numpy(weights_dict[f"layer_{i}_weights"])
            layer.biases.storage.copy_from_numpy(weights_dict[f"layer_{i}_biases"])
    
    print(f"Model weights loaded from {filename}")