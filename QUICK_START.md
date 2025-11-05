# 🚀 QUICK START: SPINN Training (Copy-Paste These Cells)

**Open Google Colab → Paste these cells in order → Run all**

---

## Cell 1: Check GPU
```python
!nvidia-smi
```

---

## Cell 2: Setup & Pull Latest Code
```python
import os

if not os.path.exists('/content/SPINN'):
    !git clone https://github.com/krithiks4/SPINN.git
    
%cd /content/SPINN
!git pull origin main
!git log --oneline -5
```

---

## Cell 3: Install Dependencies
```python
!pip install -q torch torchvision torchaudio pandas numpy matplotlib scipy scikit-learn

import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
print(f"✅ Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## Cell 4: Verify Data Files
```python
import pandas as pd
import os

files = [
    'data/processed/train.csv',
    'data/processed/val.csv', 
    'data/processed/test.csv',
    'results/checkpoints/dense_pinn_improved_final.pt'
]

for f in files:
    print(f"{'✅' if os.path.exists(f) else '❌'} {f}")

if os.path.exists('data/processed/train.csv'):
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    test = pd.read_csv('data/processed/test.csv')
    print(f"\n📊 Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
```

---

## Cell 5: (Optional) Verify Pruning Math
```python
!python verify_pruning_math.py
```

---

## Cell 6: 🚀 RUN SPINN TRAINING (45-60 minutes)
```python
!python train_spinn.py
```

**Watch for:**
- Stage 1: ~465k params (30% pruned)
- Stage 2: ~325k params (51% pruned) ← Key milestone
- Stage 3: ~260k params (61% pruned)
- Stage 4: ~209k params (69% pruned) ✅ TARGET

---

## Cell 7: Verify Success
```python
import torch
import json

# Check model
state_dict = torch.load('results/checkpoints/spinn_final.pt', map_location='cpu')
total = sum(p.numel() for p in state_dict.values())
nonzero = sum(torch.count_nonzero(p).item() for p in state_dict.values())
sparsity = 1 - (nonzero/total)

print("="*60)
print(f"Total params: {total:,}")
print(f"Non-zero params: {nonzero:,}")
print(f"Sparsity: {sparsity*100:.2f}%")

# Check metrics
with open('results/metrics/spinn_metrics.json', 'r') as f:
    m = json.load(f)

print(f"\nOverall R²: {m['final_metrics']['overall']['r2']:.4f}")
print(f"Tool R²: {m['final_metrics']['per_output']['tool_wear']['r2']:.4f}")
print(f"Thermal R²: {m['final_metrics']['per_output']['thermal_displacement']['r2']:.4f}")

print(f"\n✅ Reduction ≥ 68%: {'PASS ✅' if sparsity >= 0.68 else 'FAIL ❌'}")
print(f"✅ R² > 0.75: {'PASS ✅' if m['final_metrics']['overall']['r2'] > 0.75 else 'FAIL ❌'}")
```

---

## Cell 8: View Results
```python
from IPython.display import Image, display
display(Image('results/figures/spinn_pruning_progression.png'))
```

---

## Cell 9: Compare Dense vs SPINN
```python
import json

with open('results/metrics/improved_baseline_metrics.json') as f:
    baseline = json.load(f)
with open('results/metrics/spinn_metrics.json') as f:
    spinn = json.load(f)

print("DENSE PINN vs SPINN")
print("-"*50)
print(f"Parameters:  666,882 → {nonzero:,} ({sparsity*100:.1f}% reduction)")
print(f"Model Size:  2.55 MB → {nonzero*4/1024/1024:.2f} MB")
print(f"\nOverall R²:  {baseline['overall']['r2']:.4f} → {spinn['final_metrics']['overall']['r2']:.4f}")
print(f"Tool R²:     {baseline['per_output']['tool_wear']['r2']:.4f} → {spinn['final_metrics']['per_output']['tool_wear']['r2']:.4f}")
print(f"Thermal R²:  {baseline['per_output']['thermal_displacement']['r2']:.4f} → {spinn['final_metrics']['per_output']['thermal_displacement']['r2']:.4f}")
```

---

## Cell 10: Download Results
```python
from google.colab import files

files.download('results/checkpoints/spinn_final.pt')
files.download('results/metrics/spinn_metrics.json')
files.download('results/figures/spinn_pruning_progression.png')

print("✅ Downloaded 3 files")
```

---

## 🎯 SUCCESS CRITERIA

After Cell 7, you should see:
- ✅ **Sparsity: 68-72%** (~209k params)
- ✅ **Overall R²: > 0.75**
- ✅ **Tool R²: > 0.70**
- ✅ **Thermal R²: > 0.70**

If all pass: **PHASE 3 COMPLETE!** 🎉

Report back with final numbers and we'll move to Phase 3b (Physics Loss).

---

## 🚨 Quick Troubleshooting

**No GPU?** Runtime → Change runtime type → T4 GPU

**Files missing?** Upload your processed data and trained model to `/content/SPINN/`

**Still 15% sparsity?** Make sure Cell 2 shows commit `6c78441` in the log

**Training crashes?** Restart runtime, reduce batch_size to 256 in train_spinn.py

---

**Total time: ~1 hour** | **Start with Cell 1!** 🚀
