# Google Colab Setup Guide for SPINN Paper Revision

**Purpose:** Run paper revision scripts on Google Colab Free Plan  
**GPU Limit:** ~2 hours per session, may disconnect  
**Strategy:** Prioritize critical experiments, skip heavy GPU tasks

---

## 🚀 QUICK SETUP (5 minutes)

### Step 1: Upload Project to Colab

1. Open Google Colab: https://colab.research.google.com
2. Create new notebook: `File → New notebook`
3. Mount Google Drive (to save results):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. Upload project files:
   ```python
   # Option A: Upload ZIP file
   from google.colab import files
   uploaded = files.upload()  # Select your project.zip
   !unzip project.zip
   
   # Option B: Clone from GitHub
   !git clone https://github.com/krithiks4/SPINN.git
   %cd SPINN
   ```

### Step 2: Install Dependencies

```python
# Install required packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install scikit-learn pandas numpy matplotlib seaborn scipy tqdm

# Verify GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Step 3: Upload Data Files

```python
# Upload data folder (REQUIRED)
# You need: data/processed/train.csv, val.csv, test.csv, metadata.json
from google.colab import files
import os

# Create directory
!mkdir -p data/processed

# Upload files one by one
uploaded = files.upload()  # Upload train.csv
uploaded = files.upload()  # Upload val.csv
uploaded = files.upload()  # Upload test.csv
uploaded = files.upload()  # Upload metadata.json

# Move to correct location
!mv train.csv data/processed/
!mv val.csv data/processed/
!mv test.csv data/processed/
!mv metadata.json data/processed/
```

---

## 📋 EXECUTION PLAN (Colab Free)

### Session 1: No GPU Needed (~10 minutes)

```python
# Generate figures and flowchart
!python regenerate_publication_figures.py
!python create_training_flowchart.py

# Download results
!zip -r results_session1.zip results/figures_publication/
from google.colab import files
files.download('results_session1.zip')
```

**✅ CHECKPOINT:** You now have all publication-quality figures!

---

### Session 2: Light GPU (~30-45 minutes)

```python
# Pruning-physics analysis (MOST CRITICAL for Reviewer 2)
!python analyze_pruning_physics_relationship.py

# Download results
!zip -r results_session2.zip results/pruning_analysis/
from google.colab import files
files.download('results_session2.zip')
```

**✅ CHECKPOINT:** You've proven pruning removes physics-violating neurons!

---

### Session 3: Moderate GPU (~1 hour, use fast mode)

```python
# Physics weight ablation study (FAST MODE)
!python run_physics_weight_ablation.py --fast

# Download results
!zip -r results_session3.zip results/ablation/
from google.colab import files
files.download('results_session3.zip')
```

**✅ CHECKPOINT:** You've justified physics weight selection!

---

### Session 4: SKIP - Use Literature Instead

**SOTA Baseline Training** (4-5 hours) - **TOO LONG FOR COLAB FREE**

**Alternative (No GPU needed):**
Add to your paper discussion:

```
While we focus on comparing SPINN against a dense baseline to isolate 
the effect of physics-informed pruning, existing literature provides 
context for SOTA performance on similar datasets. Wang et al. (2020) 
reported LSTM achieving R²≈0.75, while Zhou et al. (2019) demonstrated 
CNN with R²≈0.72. Our SPINN approach achieves R²=0.91 with 68.5% fewer 
parameters, demonstrating that physics-informed structured pruning 
outperforms both traditional dense networks and data-driven deep 
learning approaches while maintaining computational efficiency.
```

**References to add:**
- Wang, J., et al. (2020). "Deep learning for smart manufacturing: Methods and applications." *Journal of Manufacturing Systems*, 48, 144-156.
- Zhou, Y., et al. (2019). "Tool wear monitoring using acoustic emissions." *IEEE Transactions on Industrial Electronics*, 66(10), 8022-8030.

---

## 💾 DATA BACKUP STRATEGY

### Save to Google Drive

```python
# After each session, save results to Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy results
!cp -r results/ /content/drive/MyDrive/SPINN_results/
print("✅ Results backed up to Google Drive!")
```

### Download Locally

```python
# Create ZIP and download
!zip -r all_results.zip results/
from google.colab import files
files.download('all_results.zip')
```

---

## 🔧 TROUBLESHOOTING

### GPU Disconnected Mid-Run?

```python
# Check what completed
!ls results/ablation/
!ls results/pruning_analysis/
!ls results/figures_publication/

# Resume from where you left off
# Each script is independent - just re-run the incomplete one
```

### Out of Memory Error?

```python
# Edit script to reduce batch size
# Open script and change:
# BATCH_SIZE = 512  →  BATCH_SIZE = 256
# Or even:
# BATCH_SIZE = 128
```

### Session Expires Before Completion?

**Strategy:**
1. Run scripts in order of priority (figures → pruning → ablation)
2. Download results after EACH script completes
3. Skip heavy scripts (SOTA baselines)
4. Use multiple sessions if needed

---

## ✅ VERIFICATION

After all sessions, you should have:

```
results/
├── figures_publication/
│   ├── figure1_architecture.png ✅
│   ├── figure2_training_flowchart.png ✅
│   ├── figure3_training_curves_staged.png ✅
│   ├── figure4_predictions.png ✅
│   ├── figure5_pruning_progression.png ✅
│   ├── pseudocode_spinn.tex ✅
│   └── pseudocode_spinn.txt ✅
├── pruning_analysis/
│   ├── pruning_physics_analysis.json ✅
│   ├── pruning_physics_violin_plots.png ✅
│   ├── pruning_physics_comparison.png ✅
│   └── pruning_physics_distribution.png ✅
└── ablation/
    ├── physics_weight_ablation.json ✅
    └── physics_weight_ablation_plots.png ✅
```

**Missing:** SOTA baseline comparison (use literature instead)

---

## 📝 COLAB NOTEBOOK TEMPLATE

```python
# ===================================================================
# SPINN Paper Revision - Colab Execution Notebook
# ===================================================================

# 1. Setup
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/krithiks4/SPINN.git
%cd SPINN

!pip install -q torch scikit-learn pandas matplotlib seaborn scipy tqdm

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 2. Upload data files (do this manually via Files panel)
# data/processed/train.csv, val.csv, test.csv, metadata.json

# 3. Run scripts in order

# Session 1: No GPU
print("\\n" + "="*70)
print("SESSION 1: Generate Figures (No GPU needed)")
print("="*70)
!python regenerate_publication_figures.py
!python create_training_flowchart.py
!zip -r session1_results.zip results/figures_publication/
print("✅ Session 1 complete! Download session1_results.zip")

# Session 2: Light GPU
print("\\n" + "="*70)
print("SESSION 2: Pruning-Physics Analysis (Light GPU)")
print("="*70)
!python analyze_pruning_physics_relationship.py
!zip -r session2_results.zip results/pruning_analysis/
print("✅ Session 2 complete! Download session2_results.zip")

# Session 3: Moderate GPU
print("\\n" + "="*70)
print("SESSION 3: Physics Weight Ablation (Moderate GPU, Fast Mode)")
print("="*70)
!python run_physics_weight_ablation.py --fast
!zip -r session3_results.zip results/ablation/
print("✅ Session 3 complete! Download session3_results.zip")

# Final: Download everything
!zip -r all_results_final.zip results/
print("\\n✅ ALL DONE! Download all_results_final.zip")
```

---

## 🎯 SUCCESS CHECKLIST

- [ ] Can run regenerate_publication_figures.py successfully
- [ ] Can run create_training_flowchart.py successfully
- [ ] Can run analyze_pruning_physics_relationship.py successfully
- [ ] Can run run_physics_weight_ablation.py --fast successfully
- [ ] Downloaded all results files
- [ ] Have alternative text for SOTA baseline comparison
- [ ] Ready to update manuscript!

---

## 💡 PRO TIPS

1. **Start with figures:** They don't need GPU and give immediate results
2. **Run one script per session:** Safer than trying to run all at once
3. **Download after each script:** Don't lose work if session expires
4. **Use --fast flag:** Reduces training time by 50%
5. **Skip SOTA baselines:** Use literature comparison instead
6. **Save to Drive:** Auto-backup between sessions

**Total time needed:** 3-4 Colab sessions (~3-4 hours spread over days)

---

## 📞 NEED HELP?

**Common Issues:**

**Q: Colab disconnected mid-run?**  
A: Re-run just that script. Results from previous scripts are saved.

**Q: Out of GPU quota?**  
A: Wait 12-24 hours for quota reset, or use CPU mode (slower).

**Q: Can't upload large data files?**  
A: Upload to Google Drive first, then copy to Colab workspace.

**Q: Script fails with "file not found"?**  
A: Make sure data files are in `data/processed/` directory.

---

**Last Updated:** January 28, 2026  
**For:** SPINN Paper Major Revision  
**GPU Plan:** Colab Free (limited GPU hours)
