# Online Adaptation Quick Start Guide

## 🎯 Goal
Implement and validate online learning capability for Gap 5, proving that your pruned SPINN model can adapt to new data with reduced computational cost.

## 📋 Prerequisites

### Before You Start:
1. ✅ Have trained models saved:
   - `models/saved/dense_pinn.pth` (Dense model)
   - `models/saved/spinn_structured.pth` or `spinn_structured_77pct.pth` (Pruned model)

2. ✅ Have processed data:
   - `data/processed/train.csv`
   - `data/processed/val.csv`
   - `data/processed/test.csv`

3. ✅ Environment ready:
   - PyTorch installed
   - GPU available (recommended) or CPU

## 🚀 Step-by-Step Execution

### Open Your Notebook
```bash
# In VS Code or Jupyter
Open: PAPER_ACTION_PLAN.ipynb
Navigate to: "🔄 ONLINE ADAPTATION IMPLEMENTATION" section
```

### Run Cells in Order:

1. **STEP 1** - Verify Setup
   - Checks if you're in correct directory
   - Verifies files exist

2. **STEP 2** - Load Libraries & Data
   - Imports PyTorch, pandas, etc.
   - Loads preprocessed CSV files

3. **STEP 3** - Load Models
   - Loads your trained Dense PINN
   - Loads your trained SPINN (pruned)
   - Verifies parameter counts

4. **STEP 4** - Prepare Data Batches
   - Splits test set into 5 batches
   - Simulates incremental data arrival

5. **STEP 5** - Define Freezing Functions
   - Sets up layer freezing utilities
   - Quick execution (<1 second)

6. **STEP 6A** - Define Training Function
   - Creates fine-tuning function
   - Quick execution (<1 second)

7. **STEP 6B** - 🔥 **RUN EXPERIMENT** (MAIN STEP)
   - Runs 3 scenarios:
     * Baseline (no adaptation)
     * Full retraining
     * Online adaptation
   - ⏱️ Takes 3-5 minutes
   - Watch progress in output

8. **STEP 7A** - Analyze Results
   - Calculates computational savings
   - Prints comparison table

9. **STEP 7B** - Generate Figures
   - Creates 4-panel visualization
   - Saves to `results/figures/`

10. **STEP 7C** - Save Results
    - Exports JSON data
    - Prints paper-ready summary

## ⏱️ Expected Timeline

| Step | Time | Notes |
|------|------|-------|
| Setup & Loading (1-3) | ~30 sec | Quick data loading |
| Preparation (4-6A) | ~10 sec | Function definitions |
| **Main Experiment (6B)** | **3-5 min** | Training loop (GPU) |
| Analysis & Viz (7A-C) | ~30 sec | Results processing |
| **TOTAL** | **5-10 min** | Depends on GPU |

## 📊 What You'll Get

### Generated Files:
1. `results/online_adaptation_results.json`
   - Complete experimental data
   - All metrics and timings

2. `results/figures/online_adaptation_analysis.png`
   - 4-panel comparison figure:
     * R² progression across batches
     * Cumulative training time
     * Parameter efficiency
     * Computational cost breakdown

### Key Metrics You'll Measure:
- **Computational Efficiency**: X% of full retraining cost
- **Time Savings**: Full retraining vs online adaptation
- **Parameter Reduction**: Trainable params (full vs frozen)
- **Accuracy Maintained**: R² comparison

## 🎯 For Your Paper

### Abstract (Add This Sentence):
> "Online adaptation experiments demonstrate that the pruned model can be fine-tuned on new cutting data with [X]% of the computational cost of full retraining, enabling efficient model updates in production environments."

Replace [X] with your actual result (likely 10-20%).

### New Table to Add:
```
Online Adaptation Performance

Approach              | R²      | Time (s) | Trainable Params | Cost vs Full
----------------------|---------|----------|------------------|-------------
Baseline (No Adapt)   | 0.97XX  | 0.00     | 0                | -
Full Retraining       | 0.97XX  | XX.XX    | 219,207          | 100%
Online Adaptation     | 0.97XX  | X.XX     | ~33,000          | ~15%
```

### New Figure to Add:
- **Figure X**: Online adaptation analysis (4-panel)
- **Caption**: "Comparison of online adaptation strategies showing (a) R² score progression, (b) cumulative training time, (c) trainable parameter efficiency, and (d) normalized computational cost across metrics."

## 🔧 Troubleshooting

### Issue: Model Files Not Found
**Solution**: Check file paths in STEP 3
```python
# Update these lines if needed:
spinn_paths = ['models/saved/spinn_structured.pth', 
               'models/saved/spinn_structured_77pct.pth']
```

### Issue: Out of Memory (GPU)
**Solution 1**: Reduce epochs
```python
# In STEP 6B, change:
NUM_EPOCHS = 10  # to → NUM_EPOCHS = 5
```

**Solution 2**: Reduce batches
```python
# In STEP 4, change:
num_batches = 5  # to → num_batches = 3
```

**Solution 3**: Use CPU (slower)
```python
# In STEP 2, force CPU:
device = torch.device('cpu')
```

### Issue: Training Too Slow
**Expected**: 3-5 min on GPU, 10-15 min on CPU
**If slower**: Reduce NUM_EPOCHS or num_batches (see above)

## ✅ Success Criteria

You've succeeded when you see:

1. ✅ All 3 scenarios complete without errors
2. ✅ Printed summary shows:
   - Online adaptation R² ≈ Full retraining R²
   - Online adaptation time < 20% of full retraining
   - Parameter reduction ≈ 85%
3. ✅ Figure saved to `results/figures/`
4. ✅ JSON results saved to `results/`

## 📝 What to Do Next

1. **Review Results**
   - Check if online adaptation maintains accuracy
   - Note the computational efficiency percentage

2. **Update Your Abstract**
   - Use VERSION 1 from revised abstract section
   - Fill in [X]% with your actual result

3. **Add to Paper**
   - Include new table in Results section
   - Add 4-panel figure
   - Mention in Discussion

4. **Update Timeline**
   - Mark Gap 5 as COMPLETE ✅
   - Move to next day's tasks

## 🎉 You're Ready!

Open `PAPER_ACTION_PLAN.ipynb` and start running cells from the "🔄 ONLINE ADAPTATION IMPLEMENTATION" section!

**Time to complete**: ~10 minutes  
**Difficulty**: Easy (just run cells sequentially)  
**Impact**: Strengthens your paper significantly! 🚀
