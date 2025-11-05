"""
Quick verification script to understand the pruning math
Run this to see why your first attempt only got 15% sparsity
"""

print("="*70)
print("CUMULATIVE PRUNING MATH EXPLAINED")
print("="*70)

# Original incorrect schedule
print("\n❌ FIRST ATTEMPT (BROKEN):")
print("Schedule: [20%, 20%, 15%, 15%]")
print("\nWhat SHOULD have happened:")
remaining = 1.0
for i, prune_ratio in enumerate([0.20, 0.20, 0.15, 0.15], 1):
    remaining = remaining * (1 - prune_ratio)
    sparsity = 1 - remaining
    print(f"  Stage {i}: Prune {prune_ratio*100:.0f}% → {remaining*100:.1f}% remain → {sparsity*100:.1f}% cumulative sparsity")

print(f"\n  Expected final sparsity: {(1-remaining)*100:.1f}%")
print(f"  Expected params: {666882 * remaining:,.0f} / 666,882")

print("\n  What ACTUALLY happened:")
print("  ⚠️  Each stage recomputed threshold on remaining weights only")
print("  ⚠️  Masks weren't preserved between stages")
print("  ⚠️  Result: Only 15% sparsity (567k params remain)")

# New corrected schedule
print("\n" + "="*70)
print("✅ FIXED VERSION:")
print("Schedule: [30%, 30%, 20%, 20%]")
print("\nCumulative calculation:")
remaining = 1.0
for i, prune_ratio in enumerate([0.30, 0.30, 0.20, 0.20], 1):
    remaining = remaining * (1 - prune_ratio)
    sparsity = 1 - remaining
    params = 666882 * remaining
    print(f"  Stage {i}: Prune {prune_ratio*100:.0f}% → {remaining*100:.1f}% remain → {sparsity*100:.1f}% sparsity → {params:,.0f} params")

print(f"\n  ✅ Final sparsity: {(1-remaining)*100:.1f}%")
print(f"  ✅ Final params: {666882 * remaining:,.0f} / 666,882")
print(f"  ✅ Reduction: {(1-remaining)*100:.1f}% (target: 70%)")

print("\n" + "="*70)
print("KEY FIXES:")
print("="*70)
print("1. Masks preserved: existing_masks parameter passed between stages")
print("2. Threshold computed: Only on ACTIVE (non-pruned) weights")
print("3. Mask enforcement: After every optimizer step during fine-tuning")
print("4. Aggressive schedule: 30-30-20-20 instead of 20-20-15-15")
print("\nRun train_spinn.py in Colab to see these numbers in action! 🚀")
