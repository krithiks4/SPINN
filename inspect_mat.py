from scipy.io import loadmat
import numpy as np

# Load the .mat file
data = loadmat('data/raw/nasa/mill.mat')

# Print all keys
print("Keys in .mat file:")
mat_keys = [k for k in data.keys() if not k.startswith('__')]
for key in mat_keys:
    print(f"  {key}")

print("\nDetailed structure:")
for key in mat_keys:
    var = data[key]
    print(f"\n{key}:")
    print(f"  Type: {type(var)}")
    if hasattr(var, 'shape'):
        print(f"  Shape: {var.shape}")
        print(f"  Dtype: {var.dtype}")
        if len(var.shape) <= 2 and var.size < 100:
            print(f"  Sample data:\n{var}")
