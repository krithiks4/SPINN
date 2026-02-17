from scipy.io import loadmat
import numpy as np

mat_data = loadmat('data/raw/nasa/mill.mat')

print("="*70)
print("NASA MILL.MAT STRUCTURE EXPLORATION")
print("="*70)

print("\nTop-level keys:")
for key in mat_data.keys():
    if not key.startswith('__'):
        value = mat_data[key]
        print(f"  {key}: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")

if 'mill' in mat_data:
    mill = mat_data['mill']
    print(f"\n'mill' structure:")
    print(f"  Type: {type(mill)}")
    print(f"  Shape: {mill.shape}")
    print(f"  dtype: {mill.dtype}")

    print(f"\nFirst case (mill[0,0]):")
    first_case = mill[0, 0]
    print(f"  Type: {type(first_case)}")
    print(f"  dtype names: {first_case.dtype.names if hasattr(first_case.dtype, 'names') else 'N/A'}")
    if hasattr(first_case.dtype, 'names'):
        print(f"\n  Fields in first case:")
        for name in first_case.dtype.names:
            field = first_case[name]
            print(f"    {name}: shape={field.shape if hasattr(field, 'shape') else 'N/A'}, type={type(field)}")

            if name in ['case', 'VB', 'DOC', 'feed']:
                try:
                    print(f"      Sample value: {field}")
                except:
                    print(f"      (complex structure)")

    print(f"\nSampling different cases:")
    for idx in [0, 10, 50, 100, 150]:
        if idx < mill.shape[1]:
            case = mill[0, idx]
            print(f"\n  Case {idx}:")
            if hasattr(case.dtype, 'names') and 'VB' in case.dtype.names:
                vb = case['VB']
                print(f"    VB (tool wear): {vb}")
            if hasattr(case.dtype, 'names') and 'case' in case.dtype.names:
                case_num = case['case']
                print(f"    case number: {case_num}")

print("\n" + "="*70)
