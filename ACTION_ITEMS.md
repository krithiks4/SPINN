# What You Need To Do

## IMMEDIATE ACTION ITEMS (Before I Start Coding)

### 1. ✅ Decision: Dataset Choice
**YOU NEED TO DECIDE:**
- [ ] **Primary dataset**: NASA Milling Dataset (recommended) or PHM Society 2010?
- [ ] **Backup plan**: Should I implement synthetic data generation if datasets are insufficient?

**My recommendation**: Start with NASA dataset. I'll build the data loader to handle it.

---

### 2. ✅ Hardware Information Needed
**PLEASE PROVIDE:**
- [ ] Your laptop CPU model (e.g., Intel i7-12700H, AMD Ryzen 9 5900HS)
- [ ] RAM amount (e.g., 16GB, 32GB)
- [ ] GPU model if available (e.g., NVIDIA RTX 3060, or "None" if CPU only)
- [ ] Operating system (Windows/Mac/Linux)

**Why**: Need this for paper's experimental setup section and to set realistic inference time targets.

---

### 3. ✅ Framework Preference
**YOU NEED TO DECIDE:**
- [ ] **PyTorch** (my recommendation - better for research, easier PINN implementation)
- [ ] **TensorFlow/Keras** (alternative if you prefer)

**My recommendation**: PyTorch for flexibility with custom physics loss functions.

---

### 4. ⚠️ Computational Resources Check
**CONFIRM:**
- [ ] Can you run Python training jobs for several hours? (needed for Phase 2-3)
- [ ] Any GPU access through cloud platforms (Google Colab, AWS, etc.)? Or laptop only?

**Note**: Training dense PINN might take 2-6 hours on laptop CPU. With GPU: 30-60 minutes.

---

## OPTIONAL BUT HELPFUL

### 5. Paper Deadline
- [ ] When is ASME MSEC submission deadline?
- [ ] Any target date for having results ready?

**Why**: Helps me prioritize what to build first.

---

### 6. Existing Resources
**DO YOU HAVE:**
- [ ] Any existing code from previous projects?
- [ ] Any literature references you want me to cite?
- [ ] Any specific ASME MSEC formatting requirements?

---

## WHAT I WILL DO AUTOMATICALLY (No Input Needed)

✅ Create complete project structure
✅ Implement data download and preprocessing
✅ Build Dense PINN architecture with physics losses
✅ Implement pruning algorithms for SPINN
✅ Create training pipelines
✅ Build validation and benchmarking scripts
✅ Generate all visualization code
✅ Set up experiment tracking
✅ Create detailed documentation

---

## SUMMARY: Minimum You Must Provide

**To start immediately, I just need:**
1. **Dataset choice**: NASA or PHM? (or both?)
2. **Laptop specs**: CPU, RAM, GPU
3. **Framework**: PyTorch or TensorFlow?

**Everything else I can proceed with defaults and best practices!**

---

## Response Template (Copy and Fill This Out)

```
DATASET: Both (NASA + PHM) - NO SYNTHETIC

HARDWARE:
- CPU: Intel Core i9 (ASUS ZenBook)
- RAM: [Will detect from system]
- GPU: Integrated Intel Iris Xe (likely) or None
- OS: Windows

FRAMEWORK: PyTorch (Selected)

DEADLINE: Paper due in 10 days - Results needed in 7 days
COMPUTE: Can run 24/7, Jupyter Notebook + Google Colab available

STATUS: ✅ CONFIRMED - BUILDING NOW
```

**Reply with the above, and I'll start building immediately!**
