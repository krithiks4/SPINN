# SPINN Paper Major Revisions - TODO List
**Generated:** January 28, 2026  
**Status:** Major Revisions Required  
**Target:** Acceptance after addressing all reviewer comments

---

## 📋 REVIEWER 1 COMMENTS - CRITICAL ISSUES

### 1. Math Notation Inconsistencies ⚠️ CRITICAL
**Issue:** Font size and styles inconsistent in equations (e.g., Eq. 2)
- [ ] Review all equations for consistent font sizes
- [ ] Ensure consistent math symbols (vectors bold, scalars normal)
- [ ] Check subscripts and superscripts formatting
- [ ] Verify LaTeX compilation of all equations
- [ ] Standardize notation across paper (e.g., L_total vs L_{total})

### 2. Loss Function Justification ⚠️ CRITICAL
**Issue:** Need to justify the SPINN loss function
- [ ] Add theoretical justification for physics-informed loss
- [ ] Explain why physics constraints improve generalization
- [ ] Reference literature on physics-informed neural networks
- [ ] Discuss trade-offs between data loss and physics loss
- [ ] Provide intuition for loss function design

### 3. Equation 1 Explanation Too Brief ⚠️ CRITICAL
**Issue:** Explanation for Eq. 1 too brief; need to explain how L1 and L2 facilitate sparsity
- [ ] Expand explanation of sparsity-inducing regularization
- [ ] Explain L1 (Lasso) promotes exactly zero weights
- [ ] Explain L2 (Ridge) prevents overfitting
- [ ] Add mathematical intuition (L1 creates sharp corners at zero)
- [ ] Reference optimization literature on sparsity
- [ ] Possibly add supplementary figure showing L1 vs L2 contours

### 4. Training Flowchart/Pseudocode Missing ⚠️ CRITICAL
**Issue:** Need flowchart or pseudo code for SPINN model training
- [ ] Create comprehensive training flowchart
- [ ] Include: initialization → dense training → pruning rounds → fine-tuning
- [ ] Add pseudocode for structured pruning algorithm
- [ ] Show decision points (convergence criteria, pruning thresholds)
- [ ] Illustrate physics loss computation
- [ ] Make it publication-quality (high resolution)

### 5. Figure Quality Issues ⚠️ CRITICAL
**Issue:** Multiple figures have small font sizes or low resolutions
- [ ] Regenerate all figures at publication quality (300 DPI minimum)
- [ ] Increase font sizes in legends (minimum 10pt)
- [ ] Increase axis label font sizes (minimum 11pt)
- [ ] Increase title font sizes (minimum 12pt)
- [ ] Ensure all text is readable when printed
- [ ] Use consistent color schemes across figures
- [ ] Export as high-resolution PDF or PNG

### 6. Figure 3 Loss Curves Need Separation ⚠️ CRITICAL
**Issue:** Magnitudes of two loss curves not at same level; recommend separating into 2-3 stages
- [ ] Separate loss plot into epochs 0-40, 41-80, 81+
- [ ] Use subplots or separate figures
- [ ] Show clear contrast between train and validation losses
- [ ] Identify any underfitting or overfitting regions
- [ ] Add zoom-in panels for different training phases
- [ ] Annotate key events (pruning rounds, convergence)

---

## 📋 REVIEWER 2 COMMENTS - SCIENTIFIC RIGOR

### 7. Motivation and References Weak ⚠️ CRITICAL
**Issue:** Motivation and significance need stronger support through additional targeted references
- [ ] Add more literature review on tool wear prediction
- [ ] Reference state-of-the-art methods in manufacturing AI
- [ ] Cite recent physics-informed neural network papers
- [ ] Discuss model compression in industrial applications
- [ ] Add references on edge computing in manufacturing
- [ ] Strengthen industrial relevance arguments
- [ ] Target: 5-10 additional high-quality references

### 8. Physics Loss Weights Not Justified ⚠️ CRITICAL
**Issue:** Clarify how default weights in combined physics loss were determined
- [ ] Add ablation study on physics loss weights
- [ ] Show sensitivity analysis (how performance changes with weights)
- [ ] Justify weight selection (grid search, validation performance)
- [ ] Discuss balance between physics fidelity and predictive accuracy
- [ ] Add supplementary table with weight variations
- [ ] Reference hyperparameter tuning methodology

### 9. Pruning-Physics Relationship Not Demonstrated ⚠️ CRITICAL
**Issue:** Claim that pruning removes neurons that violate physics needs explicit demonstration
- [ ] Add analysis of pruned neurons vs physics constraints
- [ ] Show which neurons violate physics before pruning
- [ ] Demonstrate physics loss decreases after pruning specific neurons
- [ ] Add visualization (e.g., neuron activation vs physics violation)
- [ ] Provide statistical evidence (correlation between pruning and physics adherence)
- [ ] This may require new experiments or detailed analysis

### 10. Baseline Comparison Insufficient ⚠️ CRITICAL
**Issue:** Dense Baseline underperforms literature; justify or add state-of-the-art baseline
- [ ] Investigate why dense baseline has low performance
- [ ] Compare with published results on NASA dataset
- [ ] Add at least one state-of-the-art baseline (e.g., LSTM, Transformer, CNN)
- [ ] If dense is intentionally simple, justify this choice clearly
- [ ] Show SPINN outperforms both simple and SOTA baselines
- [ ] Add comparison table with literature results

### 11. Figure Readability Issues ⚠️ CRITICAL
**Issue:** Small font sizes in legends and axis labels (duplicate of #5)
- [ ] Same as item #5 - ensure all figures are readable
- [ ] Pay special attention to legends
- [ ] Pay special attention to axis labels

---

## 📋 GENERAL IMPROVEMENTS - SCIENTIFIC CONTRIBUTION

### 12. Deeper Physical Validation Needed ⚠️ HIGH PRIORITY
**Issue:** Scientific contribution would be strengthened by deeper physical validation
- [ ] Add physics-based metrics (not just R²)
- [ ] Verify predictions follow Archard's wear law
- [ ] Check thermal predictions follow heat transfer principles
- [ ] Compare predicted vs expected physical trends
- [ ] Add domain expert validation (if possible)
- [ ] Show model respects conservation laws

### 13. Clearer Causal Arguments Needed ⚠️ HIGH PRIORITY
**Issue:** Need clearer causal arguments
- [ ] Distinguish correlation from causation in results
- [ ] Explain mechanistic reasons for improvements
- [ ] Discuss why physics constraints help generalization
- [ ] Clarify causal chain: pruning → physics adherence → better predictions
- [ ] Add causal diagrams if helpful

### 14. Writing Quality Improvements ⚠️ HIGH PRIORITY
**Issue:** Writing quality needs improvement
- [ ] Proofread entire paper for grammar and clarity
- [ ] Ensure consistent terminology throughout
- [ ] Improve transition sentences between sections
- [ ] Clarify technical jargon
- [ ] Ensure logical flow of arguments
- [ ] Check verb tenses (past for results, present for facts)
- [ ] Have colleague review for clarity

### 15. Technical Contribution Enhancement ⚠️ HIGH PRIORITY
**Issue:** Technical contribution needs improvement
- [ ] Clearly state novel contributions in introduction
- [ ] Highlight what is new compared to existing work
- [ ] Emphasize: structured pruning + physics-informed + manufacturing
- [ ] Add technical insights beyond just experimental results
- [ ] Discuss limitations and future work

---

## 🔧 TECHNICAL FIXES NEEDED

### 16. Dense Baseline Performance Investigation 🔍
**Current Status:** Dense R² = 0.65, SPINN R² = 0.91
**Issue:** Dense may be too low compared to literature
- [ ] Review training hyperparameters for dense model
- [ ] Try different architectures for dense baseline
- [ ] Ensure fair comparison (same data preprocessing)
- [ ] Check if overfitting/underfitting in dense model
- [ ] Consider retraining dense with better setup

### 17. Physics Loss Weight Ablation Study 🧪
**Current Status:** Using fixed weights (need to verify from code)
- [ ] Run ablation: vary physics loss weights
- [ ] Test at least 5 different weight combinations
- [ ] Plot performance vs weight values
- [ ] Find optimal balance
- [ ] Report results in supplementary material

### 18. State-of-the-Art Baseline Implementation 📊
**Needed:** At least one SOTA baseline
- [ ] Implement LSTM baseline for time-series
- [ ] Or implement CNN-based wear prediction
- [ ] Or compare with published results on same dataset
- [ ] Train with same data and evaluation protocol
- [ ] Add to results table

### 19. Pruning-Physics Analysis 🔬
**Needed:** Explicit demonstration of pruning removing physics-violating neurons
- [ ] Identify neurons with high physics loss contribution
- [ ] Track which neurons get pruned
- [ ] Show correlation: pruned neurons = high physics violation
- [ ] Visualize neuron activations before/after pruning
- [ ] Statistical test for significance

### 20. All Figures Regeneration 🎨
**Needed:** Publication-quality figures
- [ ] Figure 1: Architecture diagram (high res)
- [ ] Figure 2: Training flowchart (NEW - create this)
- [ ] Figure 3: Loss curves (separate into phases)
- [ ] Figure 4: Prediction results (increase font sizes)
- [ ] Figure 5: Pruning progression (increase font sizes)
- [ ] All figures: 300 DPI, fonts ≥ 10pt

---

## 📝 MANUSCRIPT SECTIONS TO REVISE

### Introduction
- [ ] Strengthen motivation with more references
- [ ] Clearly state contributions
- [ ] Add industrial relevance

### Methodology
- [ ] Add detailed equation explanations (Eq. 1)
- [ ] Justify loss function design
- [ ] Add training flowchart/pseudocode
- [ ] Explain physics constraints in detail

### Results
- [ ] Add SOTA baseline comparison
- [ ] Add physics validation metrics
- [ ] Add ablation study results
- [ ] Improve all figures
- [ ] Add statistical significance tests

### Discussion
- [ ] Clarify causal arguments
- [ ] Discuss why dense baseline is lower
- [ ] Explain physics-pruning relationship
- [ ] Add limitations section

### Conclusion
- [ ] Emphasize contributions clearly
- [ ] Discuss future work

---

## ✅ COMPLETION CHECKLIST

### Critical Path (Must Do)
- [ ] Items 1-11 (all reviewer critical comments)
- [ ] Item 16 (investigate dense baseline)
- [ ] Item 18 (add SOTA baseline)
- [ ] Item 19 (pruning-physics analysis)
- [ ] Item 20 (regenerate all figures)

### High Priority (Strongly Recommended)
- [ ] Items 12-15 (scientific rigor)
- [ ] Item 17 (ablation study)

### Medium Priority (Improves Paper)
- [ ] All manuscript section revisions
- [ ] Additional literature review
- [ ] Writing polish

---

## 🎯 SUCCESS CRITERIA

**Paper will be accepted if:**
1. ✅ All math notation is consistent
2. ✅ Loss function is theoretically justified
3. ✅ Training process is clearly illustrated (flowchart)
4. ✅ All figures are publication quality
5. ✅ Physics loss weights are justified (ablation study)
6. ✅ Pruning-physics relationship is demonstrated
7. ✅ At least one SOTA baseline is included
8. ✅ Deeper physical validation is provided
9. ✅ Writing quality is professional
10. ✅ Scientific contribution is clear and significant

---

## 📅 SUGGESTED TIMELINE

**Week 1-2:** Technical fixes (items 16-19)
**Week 3:** Figure regeneration (item 20)
**Week 4:** Manuscript revisions (items 1-15)
**Week 5:** Final polish and submission

**Total estimated time:** 4-5 weeks

---

## 🚀 NEXT STEPS - START HERE (COLAB FREE PLAN)

**Immediate actions (prioritized for limited GPU):**
1. ✅ Create this todo list
2. 🔄 Start with NO GPU NEEDED tasks:
   - Regenerate all figures at high resolution
   - Create training flowchart and pseudocode
3. 🔄 Run CRITICAL GPU tasks (light, ~2 hours):
   - Analyze pruning-physics relationship (most important!)
   - Run fast ablation study (50 epochs instead of 100)
4. ⚠️ SKIP heavy GPU tasks:
   - SOTA baseline training (use literature comparison instead)
5. 🔄 Revise manuscript addressing all comments

**Alternative for SOTA Baselines (No GPU needed):**
- Compare SPINN with published LSTM/CNN results from literature
- Justify dense baseline in discussion
- Emphasize SPINN's efficiency advantage

**Let's start!** 🎉
