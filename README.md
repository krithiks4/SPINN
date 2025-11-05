# SPINN: Sparse Physics-Informed Neural Network for CNC Milling Digital Twin

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

🎯 **Real-time tool wear and thermal displacement prediction for smart manufacturing**

---

## 📋 Project Status

**Target Conference:** ASME MSEC 2025  
**Paper Deadline:** 10 days from start  
**Status:** 🚧 Implementation in progress

### Key Metrics to Achieve:
- ✅ **70% parameter reduction** through structured pruning
- ✅ **<2% prediction error** on tool wear & thermal displacement
- ✅ **Sub-100ms inference** on consumer hardware
- ✅ **500 cycle validation** for long-term accuracy
- ✅ **15% computational cost** for online adaptation

---

## 🚀 Quick Start (3 Steps)

### Step 1: Download Datasets (DO THIS FIRST!)

#### NASA Milling Dataset (REQUIRED)
1. Go to: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
2. Find "Milling Data Set" or "Mill Tool Wear"
3. Download all CSV files
4. Place in: `data/raw/nasa/`

#### PHM 2010 Dataset (OPTIONAL - for validation)
1. Search: "PHM Society 2010 Data Challenge"
2. Download training/test data
3. Place in: `data/raw/phm/`

**See `DATASET_INSTRUCTIONS.md` for detailed steps**

### Step 2: Setup Environment

```powershell
# Create virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run Everything

#### Option A: Jupyter Notebook (Recommended for Development)
```powershell
jupyter notebook 01_train_baseline.ipynb
```
**Follow the notebook step-by-step - it has all instructions!**

#### Option B: Command Line (Automated Pipeline)
```powershell
# 1. Check datasets
python data/download_data.py --check

# 2. Preprocess data
python data/preprocess.py

# 3. Train Dense PINN baseline
python experiments/train_baseline.py

# 4. Create SPINN via pruning
python experiments/train_spinn.py

# 5. Run all experiments
python experiments/validate.py
```

---

## 📁 Project Structure

```
SPINN_ASME/
├── data/
│   ├── raw/                    # Put downloaded datasets here
│   │   ├── nasa/              # NASA milling data (CSV files)
│   │   └── phm/               # PHM 2010 data (optional)
│   ├── processed/             # Auto-generated preprocessed data
│   ├── download_data.py       # Dataset checker & validator
│   └── preprocess.py          # Data preprocessing pipeline
│
├── models/
│   ├── dense_pinn.py          # Dense PINN architecture
│   ├── physics_losses.py      # Physics-informed loss functions
│   ├── pruning.py             # Structured pruning algorithms
│   └── spinn.py               # Sparse PINN model (auto-generated)
│
├── experiments/
│   ├── train_baseline.py      # Train dense PINN
│   ├── train_spinn.py         # Create & fine-tune SPINN
│   ├── validate.py            # Run all validation experiments
│   └── online_adaptation.py   # Online learning experiments
│
├── utils/
│   ├── metrics.py             # Evaluation metrics (MAPE, RMSE, R²)
│   ├── visualization.py       # Plotting functions
│   └── benchmarking.py        # Inference time & memory benchmarks
│
├── results/
│   ├── figures/               # Generated plots for paper
│   ├── metrics/               # Performance metrics (JSON)
│   └── models/                # Saved model checkpoints
│
├── notebooks/
│   ├── 01_train_baseline.ipynb      # Main training notebook
│   ├── 02_evaluate_and_prune.ipynb  # Evaluation & pruning
│   └── 03_experiments.ipynb         # All experiments
│
├── PROJECT_PLAN.md            # Complete implementation guide
├── DATASET_INSTRUCTIONS.md    # Dataset download instructions
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

---

## 🔬 Technical Details

### Architecture

**Dense PINN (Baseline):**
- Input: 10-12 features (time, forces, process params)
- Hidden: 4 layers × 256 neurons
- Output: 2 (tool wear, thermal displacement)
- Parameters: ~260,000
- Activation: tanh (better for PINNs)

**SPINN (Pruned):**
- Target: ~78,000 parameters (70% reduction)
- Pruning: Magnitude-based structured pruning
- Stages: 4 iterative pruning cycles
- Maintain: <2% accuracy degradation

### Physics-Informed Components

1. **Mass Conservation (Wear Model)**
   ```
   Archard's Equation: dV/dt = K*(F*v)/H
   ```

2. **Energy Conservation (Thermal Model)**
   ```
   Heat Generation: Q = F*v*η
   Thermal Expansion: ΔL = α*L*ΔT
   ```

3. **Momentum Conservation (Force Balance)**
   ```
   Cutting Force: F_c = K_c * A_chip
   ```

### Loss Function
```
Total Loss = MSE_Loss + λ₁*Physics_Loss + λ₂*Boundary_Loss

Where:
- MSE_Loss: Data fitting error
- Physics_Loss: Conservation law residuals
- Boundary_Loss: Initial/boundary constraints
- λ₁, λ₂: Tunable weights (0.1-1.0)
```

---

## 📊 Experiments & Metrics

### Experiment 1: Tool Wear Prediction
- **Dataset:** NASA milling (500+ cycles)
- **Metrics:** MAPE < 2%, RMSE (μm), R² > 0.95
- **Compare:** Dense PINN vs SPINN vs Baseline NN

### Experiment 2: Thermal Displacement
- **Dataset:** Derived thermal data
- **Metrics:** MAPE < 2%, physics residuals < 5%
- **Validate:** Conservation laws satisfied

### Experiment 3: Computational Efficiency
- **Hardware:** ASUS ZenBook i9 (Windows)
- **Measure:** 
  - Inference time (ms)
  - Memory footprint (MB)
  - FLOPs reduction
  - Model size

### Experiment 4: Online Adaptation
- **Setup:** Pretrain on 80%, adapt on 20%
- **Goal:** Show 15% computational cost vs full retraining
- **Measure:** Update time, accuracy retention

---

## 🖼️ Figures for Paper

The following figures will be auto-generated:

1. **Architecture Diagram** - Dense PINN → Pruning → SPINN
2. **Training Curves** - Loss evolution (data + physics)
3. **Tool Wear Prediction** - 3-panel comparison plot
4. **Thermal Deformation** - Error heatmaps & physics residuals
5. **Computational Efficiency** - Bar charts (time, params, memory)
6. **Online Adaptation** - Incremental learning performance

All saved in: `results/figures/` (publication-quality, 300 DPI)

---

## ⚙️ System Requirements

### Minimum:
- **CPU:** Modern multi-core (i5/Ryzen 5 or better)
- **RAM:** 8 GB
- **Storage:** 5 GB free space
- **OS:** Windows 10/11, macOS, or Linux
- **Python:** 3.8+

### Your Setup (ASUS ZenBook i9):
- ✅ **CPU:** Intel Core i9 (excellent for inference benchmarks)
- ✅ **RAM:** 16+ GB (assumed)
- ✅ **GPU:** Integrated Intel Iris Xe or None (CPU training is fine)
- ✅ **OS:** Windows 11
- ⏱️ **Training time:** 2-6 hours on CPU, <1 hour on GPU (via Colab)

### Cloud Options:
- **Google Colab:** Free GPU (T4), 12GB RAM - Recommended for faster training
- **Kaggle Kernels:** Free GPU, good for experiments
- **Your laptop:** Perfect for inference benchmarks & paper results

---

## 📝 Timeline (10 Days)

### Days 1-2: Data & Baseline (YOU ARE HERE)
- ✅ Download datasets
- ✅ Preprocess data
- ✅ Train Dense PINN
- ✅ Validate baseline performance

### Days 3-4: SPINN Creation
- Implement pruning
- Iterative pruning (4 stages)
- Fine-tune SPINN
- Verify 70% reduction + <2% error

### Days 5-6: Experiments
- Tool wear validation (500 cycles)
- Thermal displacement experiments
- Computational benchmarks
- Online adaptation tests

### Days 7-8: Paper Writing
- Generate all figures
- Write methods section
- Document results
- Create tables

### Days 9-10: Review & Submit
- Proofread paper
- Verify all numbers
- Format for ASME MSEC
- Submit!

---

## 🐛 Troubleshooting

### Problem: Can't find NASA dataset
**Solution:** See `DATASET_INSTRUCTIONS.md` for direct links. Dataset may have moved - search "NASA milling tool wear dataset" or email prognostics@arc.nasa.gov

### Problem: PyTorch installation fails
**Solution:** 
```powershell
# Try this instead:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Problem: Training is too slow
**Solution:** Use Google Colab with free GPU:
1. Upload notebooks to Colab
2. Change runtime to GPU (Runtime → Change runtime type)
3. Upload preprocessed data to Colab
4. Train there, download results

### Problem: Physics loss dominates/ignored
**Solution:** Tune lambda weights in `models/physics_losses.py`:
```python
lambda_physics = 0.1  # Start here
# If physics loss too high: decrease to 0.01
# If data loss too high: increase to 0.5
```

### Problem: Can't achieve 70% reduction with <2% error
**Solution:** 
- Adjust pruning schedule (prune less per stage)
- Increase fine-tuning epochs
- Try different pruning ratios per layer

---

## 📚 Key References

### Physics-Informed Neural Networks:
- Raissi et al., "Physics-informed neural networks" (2019)
- Karniadakis et al., "Physics-informed machine learning" (2021)

### Neural Network Pruning:
- Han et al., "Learning both weights and connections" (2015)
- Li et al., "Pruning filters for efficient convnets" (2017)

### Manufacturing Digital Twins:
- Liu et al., "Digital twin-driven manufacturing" (2020)
- Tao et al., "Data-driven smart manufacturing" (2018)

---

## 📧 Need Help?

1. **Check documentation:** `PROJECT_PLAN.md` has detailed explanations
2. **Dataset issues:** See `DATASET_INSTRUCTIONS.md`
3. **Code errors:** Check inline comments in Python files
4. **Jupyter notebooks:** Follow step-by-step instructions

---

## ✅ Current Status

### Completed:
✅ Project structure created  
✅ Data pipeline implemented  
✅ Dense PINN architecture  
✅ Physics loss functions  
✅ Pruning algorithms  
✅ Training notebooks  

### Your Next Steps:
1. ⬇️  **Download NASA dataset** (see `DATASET_INSTRUCTIONS.md`)
2. 📂 **Place files in** `data/raw/nasa/`
3. 🚀 **Open** `01_train_baseline.ipynb`
4. ▶️  **Run all cells** step-by-step

---

## 📄 License

MIT License - Free to use for research and publication.

---

## 🎯 Remember the Goal

**You're building a state-of-the-art sparse physics-informed neural network that:**
- Reduces parameters by 70%
- Maintains <2% error
- Enables real-time edge deployment
- Outperforms purely data-driven approaches
- Demonstrates practical smart manufacturing applications

**This is a strong contribution to ASME MSEC! Let's make it happen! 🚀**

---

**Last Updated:** November 4, 2025  
**Status:** Ready for implementation - datasets needed  
**Time to Results:** 7 days from dataset download
