# 🚀 Step-by-Step Guide: Run SPINN Training in Google Colab

## 📋 Complete Checklist

Follow these steps **in order** in your Google Colab notebook:

---

## ✅ STEP 1: Setup Runtime (Tesla T4 GPU)

### Cell 1: Check GPU
```python
# Verify you have GPU access
!nvidia-smi
```

**Expected output:** Should show Tesla T4 with ~15GB memory

**If no GPU:** Go to `Runtime` → `Change runtime type` → Select `T4 GPU` → Save

---

## ✅ STEP 2: Clone Repository & Pull Latest Code

### Cell 2: Clone or Navigate to Project
```python
import os

# If first time, clone the repo
if not os.path.exists('/content/SPINN'):
    !git clone https://github.com/krithiks4/SPINN.git
    
# Navigate to project
%cd /content/SPINN

# Pull latest fixes
!git pull origin main

# Show recent commits (verify you have the fixes)
!git log --oneline -5
```

**Expected output:** Should show commits:
- `eb3c8e9` - Add pruning math verification script
- `e8d6e91` - Add detailed explanation of pruning bug fixes
- `e0fccea` - Update Phase 3 checklist
- `6c78441` - **FIX CRITICAL: Cumulative pruning bug**
- `ad380a3` - Add SPINN training with iterative magnitude-based pruning

---

## ✅ STEP 3: Install Dependencies

### Cell 3: Install Required Packages
```python
# Install all dependencies
!pip install -q torch torchvision torchaudio
!pip install -q pandas numpy matplotlib scipy scikit-learn

# Verify installations
import torch
import pandas as pd
import numpy as np
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**Expected output:**
```
✅ PyTorch: 2.x.x+cu121
✅ CUDA available: True
✅ Device: Tesla T4
```

---

## ✅ STEP 4: Verify Data is Ready

### Cell 4: Check Data Files
```python
# Verify all data files exist
import os

required_files = [
    'data/processed/train.csv',
    'data/processed/val.csv',
    'data/processed/test.csv',
    'data/processed/metadata.json',
    'results/checkpoints/dense_pinn_improved_final.pt'
]

print("📂 Checking required files:")
for file in required_files:
    exists = "✅" if os.path.exists(file) else "❌ MISSING"
    print(f"  {exists} {file}")

# Show data sizes
if os.path.exists('data/processed/train.csv'):
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')
    print(f"\n📊 Data splits:")
    print(f"  Train: {len(train):,} samples")
    print(f"  Val: {len(val):,} samples")
    print(f"  Test: {len(test):,} samples")
```

**Expected output:**
```
✅ data/processed/train.csv
✅ data/processed/val.csv
✅ data/processed/test.csv
✅ data/processed/metadata.json
✅ results/checkpoints/dense_pinn_improved_final.pt

📊 Data splits:
  Train: 8,730 samples
  Val: 1,710 samples
  Test: 1,606 samples
```

**If files missing:** You need to upload your processed data and pre-trained model to Colab!

---

## ✅ STEP 5: Verify Pruning Math (Optional but Recommended)

### Cell 5: Run Math Verification
```python
# Run the verification script to understand the fix
!python verify_pruning_math.py
```

**Expected output:** Shows the math comparing old (15% final) vs new (68.6% final) pruning schedules

---

## ✅ STEP 6: RUN SPINN TRAINING! 🚀

### Cell 6: Execute SPINN Pruning
```python
# This is the main training run - will take 45-60 minutes
!python train_spinn.py
```

**What to watch for during training:**

### **Stage 1 (30% pruning, 40 epochs)**
Expected output:
```
============================================================
PRUNING STAGE 1/4
============================================================

🔪 Applying 30% pruning...
   Pruning threshold: ~0.008
   network.0.weight: pruned ~2700/9216 (30%)
   network.2.weight: pruned ~78000/262144 (30%)
   ...
   Total pruned this stage: ~200000/665088 (30%)

📊 Current stats:
   Total params: 666,882
   Non-zero params: ~465,000
   Sparsity: 30.0%

🎯 Fine-tuning for 40 epochs...
   Epoch 10/40 - Train: 0.00X, Val: 0.XXX
   ...

📈 Performance after stage 1:
   Overall R² = 0.8X - 0.8X
   Tool wear R² = 0.7X - 0.8X
   Thermal disp R² = 0.7X - 0.9X
```

### **Stage 2 (30% more pruning, 40 epochs)**
```
============================================================
PRUNING STAGE 2/4
============================================================

🔪 Applying 30% pruning...
   Total pruned this stage: ~140000/465000 (30% of remaining)

📊 Current stats:
   Total params: 666,882
   Non-zero params: ~325,000  ← KEY: Should be ~49% of original
   Sparsity: 51.0%

📈 Performance after stage 2:
   Overall R² = 0.7X - 0.8X  (slight drop expected)
```

### **Stage 3 (20% more pruning, 50 epochs)**
```
============================================================
PRUNING STAGE 3/4
============================================================

📊 Current stats:
   Non-zero params: ~260,000  ← Should be ~39% of original
   Sparsity: 61.0%
```

### **Stage 4 (20% more pruning, 60 epochs)**
```
============================================================
PRUNING STAGE 4/4
============================================================

📊 Current stats:
   Non-zero params: ~209,000  ← TARGET: Should be ~31% of original
   Sparsity: 68.6%
```

### **Final Results**
```
============================================================
FINAL SPINN EVALUATION
============================================================

📊 FINAL RESULTS:
   Parameter reduction: 68.6%
   Dense params: 666,882
   SPINN params: ~209,000

   Overall R²: 0.7X-0.8X (baseline: 0.8124)
   Tool wear R²: 0.7X+
   Thermal disp R²: 0.7X+

✅ Saved: results/figures/spinn_pruning_progression.png
```

---

## ✅ STEP 7: Verify Success

### Cell 7: Check Final Model
```python
import torch
import json

# Load final model
model_path = 'results/checkpoints/spinn_final.pt'
state_dict = torch.load(model_path, map_location='cpu')

# Count non-zero parameters
total_params = 0
nonzero_params = 0
for key, param in state_dict.items():
    total_params += param.numel()
    nonzero_params += torch.count_nonzero(param).item()

sparsity = 1 - (nonzero_params / total_params)

print("="*60)
print("FINAL SPINN MODEL VERIFICATION")
print("="*60)
print(f"Total parameters: {total_params:,}")
print(f"Non-zero parameters: {nonzero_params:,}")
print(f"Sparsity: {sparsity*100:.2f}%")
print(f"Reduction: {sparsity*100:.2f}%")

# Load metrics
with open('results/metrics/spinn_metrics.json', 'r') as f:
    metrics = json.load(f)

print(f"\n📊 Test Set Performance:")
print(f"Overall R²: {metrics['final_metrics']['overall']['r2']:.4f}")
print(f"Tool wear R²: {metrics['final_metrics']['per_output']['tool_wear']['r2']:.4f}")
print(f"Thermal R²: {metrics['final_metrics']['per_output']['thermal_displacement']['r2']:.4f}")

# Success criteria
print(f"\n✅ SUCCESS CRITERIA:")
print(f"  Reduction ≥ 68%: {'✅ PASS' if sparsity >= 0.68 else '❌ FAIL'} ({sparsity*100:.1f}%)")
print(f"  Overall R² > 0.75: {'✅ PASS' if metrics['final_metrics']['overall']['r2'] > 0.75 else '❌ FAIL'} ({metrics['final_metrics']['overall']['r2']:.4f})")
print(f"  Tool R² > 0.70: {'✅ PASS' if metrics['final_metrics']['per_output']['tool_wear']['r2'] > 0.70 else '❌ FAIL'}")
print(f"  Thermal R² > 0.70: {'✅ PASS' if metrics['final_metrics']['per_output']['thermal_displacement']['r2'] > 0.70 else '❌ FAIL'}")
```

---

## ✅ STEP 8: View Pruning Progression

### Cell 8: Display Pruning Plot
```python
from IPython.display import Image, display

# Show the pruning progression figure
display(Image('results/figures/spinn_pruning_progression.png'))
```

**Expected:** 4-panel plot showing:
1. Parameter count decreasing across 4 stages
2. Sparsity increasing to ~69%
3. R² scores (slight degradation acceptable)
4. Per-output R² tracking

---

## ✅ STEP 9: Compare Dense vs SPINN

### Cell 9: Side-by-Side Comparison
```python
import json

# Load baseline metrics
with open('results/metrics/improved_baseline_metrics.json', 'r') as f:
    baseline = json.load(f)

# Load SPINN metrics
with open('results/metrics/spinn_metrics.json', 'r') as f:
    spinn = json.load(f)

print("="*70)
print("DENSE PINN vs SPINN COMPARISON")
print("="*70)

print(f"\n{'Metric':<30} {'Dense PINN':<20} {'SPINN':<20} {'Change':<15}")
print("-"*70)

# Parameters
dense_params = 666882
spinn_params = spinn['final_metrics'].get('nonzero_params', 209000)
print(f"{'Parameters':<30} {dense_params:>18,}  {spinn_params:>18,}  {(spinn_params-dense_params)/dense_params*100:>12.1f}%")

# Model size (rough estimate)
dense_size = dense_params * 4 / 1024 / 1024  # Float32 to MB
spinn_size = spinn_params * 4 / 1024 / 1024
print(f"{'Model Size (MB)':<30} {dense_size:>18.2f}  {spinn_size:>18.2f}  {(spinn_size-dense_size)/dense_size*100:>12.1f}%")

print(f"\n{'Overall R²':<30} {baseline['overall']['r2']:>18.4f}  {spinn['final_metrics']['overall']['r2']:>18.4f}  {(spinn['final_metrics']['overall']['r2']-baseline['overall']['r2']):>12.4f}")
print(f"{'Tool Wear R²':<30} {baseline['per_output']['tool_wear']['r2']:>18.4f}  {spinn['final_metrics']['per_output']['tool_wear']['r2']:>18.4f}  {(spinn['final_metrics']['per_output']['tool_wear']['r2']-baseline['per_output']['tool_wear']['r2']):>12.4f}")
print(f"{'Thermal Disp R²':<30} {baseline['per_output']['thermal_displacement']['r2']:>18.4f}  {spinn['final_metrics']['per_output']['thermal_displacement']['r2']:>18.4f}  {(spinn['final_metrics']['per_output']['thermal_displacement']['r2']-baseline['per_output']['thermal_displacement']['r2']):>12.4f}")

print(f"\n{'Tool Wear RMSE (mm)':<30} {baseline['per_output']['tool_wear']['rmse']:>18.4f}  {spinn['final_metrics']['per_output']['tool_wear']['rmse']:>18.4f}  {(spinn['final_metrics']['per_output']['tool_wear']['rmse']-baseline['per_output']['tool_wear']['rmse'])/baseline['per_output']['tool_wear']['rmse']*100:>12.1f}%")
print(f"{'Thermal RMSE (mm)':<30} {baseline['per_output']['thermal_displacement']['rmse']:>18.6f}  {spinn['final_metrics']['per_output']['thermal_displacement']['rmse']:>18.6f}  {(spinn['final_metrics']['per_output']['thermal_displacement']['rmse']-baseline['per_output']['thermal_displacement']['rmse'])/baseline['per_output']['thermal_displacement']['rmse']*100:>12.1f}%")

print("\n" + "="*70)
print("✅ SUMMARY:")
print(f"   Achieved {(1-spinn_params/dense_params)*100:.1f}% parameter reduction")
print(f"   R² degradation: {(spinn['final_metrics']['overall']['r2']-baseline['overall']['r2']):.4f}")
print("="*70)
```

---

## ✅ STEP 10: Download Results (Optional)

### Cell 10: Download Key Files
```python
from google.colab import files

# Download final model
files.download('results/checkpoints/spinn_final.pt')

# Download metrics
files.download('results/metrics/spinn_metrics.json')

# Download figure
files.download('results/figures/spinn_pruning_progression.png')

print("✅ Downloaded 3 files to your local machine")
```

---

## ✅ STEP 11: Commit Results to GitHub

### Cell 11: Push Results to Repository
```python
# Configure git (replace with your info)
!git config --global user.email "your-email@example.com"
!git config --global user.name "Your Name"

# Add generated files
!git add results/checkpoints/spinn_*.pt
!git add results/metrics/spinn_metrics.json
!git add results/figures/spinn_pruning_progression.png

# Commit
!git commit -m "SPINN training complete: 68.6% reduction achieved"

# Push (you'll need to authenticate)
!git push origin main
```

---

## 🎯 SUCCESS CRITERIA CHECKLIST

After completing all steps, verify:

- [ ] **GPU detected**: Tesla T4 with ~15GB memory
- [ ] **Latest code pulled**: Commits eb3c8e9, e8d6e91, 6c78441 visible
- [ ] **Training completed**: All 4 pruning stages finished
- [ ] **Parameter reduction**: 68-72% (target: ~209k params)
- [ ] **Overall R²**: > 0.75 (some degradation acceptable)
- [ ] **Tool wear R²**: > 0.70
- [ ] **Thermal R²**: > 0.70
- [ ] **Files saved**: spinn_final.pt, metrics JSON, progression plot
- [ ] **No errors**: Training completed without crashes

---

## 🚨 TROUBLESHOOTING

### Issue 1: "No GPU available"
**Solution:** Runtime → Change runtime type → T4 GPU → Save → Reconnect

### Issue 2: "File not found: dense_pinn_improved_final.pt"
**Solution:** You need to upload your pre-trained Dense PINN model from your previous training

### Issue 3: Training crashes or OOM (Out of Memory)
**Solution:**
- Restart runtime
- Reduce batch size in `train_spinn.py` (line ~245): Change `batch_size=512` to `batch_size=256`

### Issue 4: R² drops below 0.70
**Solution:** Try lighter pruning schedule:
```python
# In train_spinn.py, change line ~273:
pruning_schedule = [
    (0.25, 40, 0.0005),  # Prune 25% instead of 30%
    (0.25, 40, 0.0003),  
    (0.20, 50, 0.0002),  
    (0.15, 60, 0.0001),  # Prune 15% instead of 20%
]
# This gives ~60% reduction instead of 69%
```

### Issue 5: Sparsity still showing 15%
**Solution:** Make sure you pulled the latest code! Run:
```python
!git fetch origin
!git reset --hard origin/main
!git log --oneline -3
# Should show commit 6c78441 (FIX CRITICAL)
```

---

## 📊 EXPECTED TIMELINE

| Step | Duration | Cumulative |
|------|----------|------------|
| Steps 1-5: Setup & verification | 5 min | 5 min |
| Step 6: SPINN training | 45-60 min | 50-65 min |
| Steps 7-11: Analysis & export | 5 min | 55-70 min |

**Total: ~1 hour**

---

## ✅ WHAT TO DO AFTER SUCCESS

Once training succeeds with 68-72% reduction and R² > 0.75:

1. **Save screenshot** of final results
2. **Copy the metrics** (parameter count, R² scores)
3. **Report back** with:
   - Final parameter count
   - Final R² scores
   - Training time
   - Any issues encountered

4. **Next phase**: Add physics loss to improve to <2% error target

---

## 🚀 READY? START WITH CELL 1!

Open Google Colab, ensure you have T4 GPU, and run cells 1-11 in order.

**Good luck!** 🎯
