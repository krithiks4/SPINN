import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

def create_training_flowchart():
    print("📊 Creating SPINN training flowchart...")
    fig, ax = plt.subplots(figsize=(14, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30)
    ax.axis('off')

    colors = {
        'start': '#27AE60',
        'data': '#3498DB',
        'model': '#E74C3C',
        'physics': '#9B59B6',
        'pruning': '#E67E22',
        'decision': '#F39C12',
        'end': '#2ECC71'
    }
    y_pos = 29
    x_center = 5

    def draw_box(ax, text, x, y, width, height, color, style='round'):
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle=f"{style},pad=0.1",
            edgecolor='black', facecolor=color,
            linewidth=2, alpha=0.8
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center',
               fontsize=11, fontweight='bold', wrap=True)
        return y - height/2
    def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=20,
            linewidth=2.5, color='black', alpha=0.7
        )
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, label, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    def draw_decision(ax, text, x, y, size, color):

        diamond = mpatches.FancyBboxPatch(
            (x - size/2, y - size/2), size, size,
            boxstyle="round,pad=0.05",
            edgecolor='black', facecolor=color,
            linewidth=2, alpha=0.8,
            transform=ax.transData
        )

        import matplotlib.transforms as transforms
        t = transforms.Affine2D().rotate_deg(45) + ax.transData
        diamond.set_transform(t)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center',
               fontsize=10, fontweight='bold')
        return y

    ax.text(x_center, y_pos, 'SPINN Training Pipeline', 
           ha='center', fontsize=18, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    y_pos -= 1.5

    y_pos = draw_box(ax, 'START', x_center, y_pos, 2, 0.8, colors['start'], 'round')
    y_pos -= 0.5
    draw_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos)
    y_pos -= 0.3

    ax.text(x_center, y_pos, 'PHASE 1: DATA PREPARATION', 
           ha='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow'))
    y_pos -= 1
    y_pos = draw_box(ax, 'Load NASA Milling\nDataset (167 experiments)', 
                    x_center, y_pos, 3, 0.8, colors['data'])
    y_pos -= 0.5
    draw_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos)
    y_pos -= 0.3
    y_pos = draw_box(ax, 'Feature Engineering\n16 features + physics', 
                    x_center, y_pos, 3, 0.8, colors['data'])
    y_pos -= 0.5
    draw_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos)
    y_pos -= 0.3
    y_pos = draw_box(ax, 'Train/Val/Test Split\n(70/15/15)', 
                    x_center, y_pos, 3, 0.8, colors['data'])
    y_pos -= 0.8
    draw_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos)
    y_pos -= 0.5

    ax.text(x_center, y_pos, 'PHASE 2: DENSE MODEL TRAINING', 
           ha='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue'))
    y_pos -= 1
    y_pos = draw_box(ax, 'Initialize Dense PINN\n[512, 512, 512, 256]\n666,882 parameters', 
                    x_center, y_pos, 3, 1.2, colors['model'])
    y_pos -= 0.7
    draw_arrow(ax, x_center, y_pos + 0.7, x_center, y_pos)
    y_pos -= 0.3

    loop_top = y_pos
    y_pos = draw_box(ax, 'Epoch Loop\n(max 200 epochs)', 
                    x_center, y_pos, 3, 0.8, colors['model'])
    y_pos -= 0.5
    draw_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos)
    y_pos -= 0.3
    y_pos = draw_box(ax, 'Compute Combined Loss:\nL = L_data + λ₁L_Archard +\nλ₂L_thermal + λ₃L_force', 
                    x_center, y_pos, 3.5, 1.2, colors['physics'])
    y_pos -= 0.7
    draw_arrow(ax, x_center, y_pos + 0.7, x_center, y_pos)
    y_pos -= 0.3
    y_pos = draw_box(ax, 'Backpropagation +\nGradient Clipping +\nOptimizer Step', 
                    x_center, y_pos, 3, 1, colors['model'])
    y_pos -= 0.6
    draw_arrow(ax, x_center, y_pos + 0.6, x_center, y_pos)
    y_pos -= 0.3

    y_pos = draw_box(ax, 'Converged?', x_center, y_pos, 2, 0.8, colors['decision'], 'round')

    draw_arrow(ax, x_center - 1, y_pos, x_center - 3.5, y_pos, 'No')
    draw_arrow(ax, x_center - 3.5, y_pos, x_center - 3.5, loop_top)
    draw_arrow(ax, x_center - 3.5, loop_top, x_center, loop_top)

    y_pos -= 0.5
    draw_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos, 'Yes')
    y_pos -= 0.5

    ax.text(x_center, y_pos, 'PHASE 3: STRUCTURED PRUNING', 
           ha='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFE5CC'))
    y_pos -= 1
    y_pos = draw_box(ax, 'Initialize Pruning\nStage = 1', 
                    x_center, y_pos, 2.5, 0.8, colors['pruning'])
    y_pos -= 0.5
    draw_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos)

    prune_loop_top = y_pos
    y_pos -= 0.3
    y_pos = draw_box(ax, 'Compute Neuron\nImportance Scores', 
                    x_center, y_pos, 2.5, 0.8, colors['pruning'])
    y_pos -= 0.5
    draw_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos)
    y_pos -= 0.3
    y_pos = draw_box(ax, 'Prune Low-Importance\nNeurons (25%)', 
                    x_center, y_pos, 2.5, 0.8, colors['pruning'])
    y_pos -= 0.5
    draw_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos)
    y_pos -= 0.3
    y_pos = draw_box(ax, 'Fine-tune Model\n(30 epochs)', 
                    x_center, y_pos, 2.5, 0.8, colors['model'])
    y_pos -= 0.5
    draw_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos)
    y_pos -= 0.3
    y_pos = draw_box(ax, 'Evaluate R²', 
                    x_center, y_pos, 2.5, 0.8, colors['model'])
    y_pos -= 0.5
    draw_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos)
    y_pos -= 0.3

    decision_y = y_pos
    y_pos = draw_box(ax, 'Stage < 4 AND\nR² > 0.75?', 
                    x_center, y_pos, 2.5, 1, colors['decision'], 'round')

    draw_arrow(ax, x_center - 1.25, y_pos, x_center - 3.5, y_pos, 'Yes')
    draw_arrow(ax, x_center - 3.5, y_pos, x_center - 3.5, prune_loop_top + 0.3)
    ax.text(x_center - 3.8, prune_loop_top + 0.5, 'Stage++', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white'))
    draw_arrow(ax, x_center - 3.5, prune_loop_top + 0.3, x_center, prune_loop_top + 0.3)

    y_pos -= 0.6
    draw_arrow(ax, x_center, decision_y - 0.5, x_center, y_pos, 'No')
    y_pos -= 0.3

    ax.text(x_center, y_pos, 'PHASE 4: EVALUATION', 
           ha='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen'))
    y_pos -= 1
    y_pos = draw_box(ax, 'Test Set Evaluation\nCompute R², RMSE, MAE', 
                    x_center, y_pos, 3, 0.8, colors['model'])
    y_pos -= 0.5
    draw_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos)
    y_pos -= 0.3
    y_pos = draw_box(ax, 'Physics Validation\nVerify conservation laws', 
                    x_center, y_pos, 3, 0.8, colors['physics'])
    y_pos -= 0.5
    draw_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos)
    y_pos -= 0.3
    y_pos = draw_box(ax, 'Save Final Model\n+ Results', 
                    x_center, y_pos, 3, 0.8, colors['model'])
    y_pos -= 0.5
    draw_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos)
    y_pos -= 0.3

    y_pos = draw_box(ax, 'END', x_center, y_pos, 2, 0.8, colors['end'], 'round')

    legend_x = 8.5
    legend_y = 28
    ax.text(legend_x, legend_y, 'Legend:', fontsize=12, fontweight='bold')
    legend_y -= 0.7
    legend_items = [
        ('Data Processing', colors['data']),
        ('Model Operations', colors['model']),
        ('Physics Loss', colors['physics']),
        ('Pruning', colors['pruning']),
        ('Decision', colors['decision'])
    ]
    for item, color in legend_items:
        draw_box(ax, '', legend_x, legend_y, 0.5, 0.4, color)
        ax.text(legend_x + 0.4, legend_y, item, fontsize=9, va='center')
        legend_y -= 0.6
    plt.tight_layout()
    output_path = 'results/figures_publication/figure2_training_flowchart.png'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {output_path}")
    plt.close()

def generate_pseudocode():
    print("\n📝 Generating training pseudocode...")
    pseudocode = r"""
\begin{algorithm}
\caption{SPINN Training Algorithm}
\begin{algorithmic}[1]
\Require Training data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$
\Require Physics loss weights $\lambda_1, \lambda_2, \lambda_3$
\Require Pruning stages $S$, pruning ratio $\rho$
\Ensure Pruned physics-informed model $f_{\theta'}$

\State \textbf{Phase 1: Dense Model Training}
\State Initialize dense network $f_\theta$ with parameters $\theta$
\For{epoch = 1 to $E_{max}$}
    \For{batch $(X, Y) \in \mathcal{D}$}
        \State $\hat{Y} \gets f_\theta(X)$ \Comment{Forward pass}
        \State $\mathcal{L}_{data} \gets \text{MSE}(\hat{Y}, Y)$
        \State $\mathcal{L}_{physics} \gets \lambda_1 \mathcal{L}_{Archard} + \lambda_2 \mathcal{L}_{thermal} + \lambda_3 \mathcal{L}_{force}$
        \State $\mathcal{L}_{total} \gets \mathcal{L}_{data} + \mathcal{L}_{physics}$
        \State $\theta \gets \theta - \alpha \nabla_\theta \mathcal{L}_{total}$ \Comment{Backpropagation}
    \EndFor
    \If{converged on validation set}
        \State \textbf{break}
    \EndIf
\EndFor

\State \textbf{Phase 2: Structured Pruning}
\For{stage $s = 1$ to $S$}
    \State Compute neuron importance scores $I_j$ for each neuron $j$
    \State $\mathcal{N}_{prune} \gets \text{bottom-}\rho \text{ neurons by } I_j$
    \State Remove neurons in $\mathcal{N}_{prune}$ from $f_\theta$ \Comment{Structured pruning}
    \State $\theta \gets$ new parameter set after pruning
    \For{epoch = 1 to $E_{finetune}$}
        \For{batch $(X, Y) \in \mathcal{D}$}
            \State Fine-tune $\theta$ with combined loss $\mathcal{L}_{total}$
        \EndFor
    \EndFor
    \State Evaluate $R^2$ on validation set
    \If{$R^2 < R^2_{min}$}
        \State \textbf{break} \Comment{Stop if performance degrades}
    \EndIf
\EndFor

\State \textbf{Phase 3: Final Evaluation}
\State Evaluate pruned model $f_{\theta'}$ on test set
\State Verify physics constraint satisfaction
\State \Return $f_{\theta'}$

\end{algorithmic}
\end{algorithm}

% Physics Loss Definitions:
% L_Archard: Archard wear equation violation
% L_thermal: Thermal expansion equation violation
% L_force: Force balance equation violation
% Importance score: I_j = ||w_j||_2 (L2 norm of neuron weights)
================================================================================
SPINN TRAINING ALGORITHM (Plain Text)
================================================================================

INPUTS:
  - Training data D = {(x_i, y_i)} for i=1 to N
  - Physics loss weights: λ₁, λ₂, λ₃
  - Pruning stages: S
  - Pruning ratio: ρ

OUTPUT:
  - Pruned physics-informed model f_θ'

--------------------------------------------------------------------------------
PHASE 1: DENSE MODEL TRAINING
--------------------------------------------------------------------------------
1. Initialize dense network f_θ with parameters θ

2. FOR epoch = 1 to E_max:
     FOR each batch (X, Y) in D:
       2.1. Forward pass: Ŷ = f_θ(X)
       2.2. Compute data loss: L_data = MSE(Ŷ, Y)
       2.3. Compute physics losses:
            L_Archard = Archard wear equation violation
            L_thermal = Thermal expansion violation
            L_force = Force balance violation
       2.4. Combined loss: L_total = L_data + λ₁L_Archard + λ₂L_thermal + λ₃L_force
       2.5. Backpropagation: θ ← θ - α∇_θ L_total
       2.6. Gradient clipping: clip(∇_θ, max_norm=1.0)
     END FOR
     IF converged on validation set:
       BREAK
     END IF
   END FOR

--------------------------------------------------------------------------------
PHASE 2: STRUCTURED PRUNING
--------------------------------------------------------------------------------
3. FOR stage s = 1 to S:
     3.1. Compute neuron importance scores:
          I_j = ||w_j||₂ (L2 norm of neuron weights)
     3.2. Identify neurons to prune:
          N_prune = bottom-ρ neurons by importance I_j
     3.3. Remove neurons N_prune from f_θ (structured pruning)
          Update parameter set θ after removal
     3.4. Fine-tuning loop:
          FOR epoch = 1 to E_finetune:
            FOR each batch (X, Y) in D:
              Fine-tune θ with combined loss L_total
            END FOR
          END FOR
     3.5. Evaluate R² on validation set
     3.6. IF R² < R²_min:
            BREAK (stop if performance degrades too much)
          END IF
   END FOR

--------------------------------------------------------------------------------
PHASE 3: FINAL EVALUATION
--------------------------------------------------------------------------------
4. Evaluate pruned model f_θ' on test set:
   - Compute R², RMSE, MAE
   - Verify physics constraint satisfaction
   - Compare with baseline models

5. RETURN pruned model f_θ'

================================================================================
KEY PARAMETERS:
================================================================================
- E_max: Maximum training epochs (e.g., 200)
- E_finetune: Fine-tuning epochs per stage (e.g., 30)
- S: Number of pruning stages (e.g., 4)
- ρ: Pruning ratio per stage (e.g., 0.25 = 25% neurons removed)
- R²_min: Minimum acceptable R² (e.g., 0.75)
- α: Learning rate (e.g., 0.001)
- λ₁, λ₂, λ₃: Physics loss weights (determined by ablation study)

================================================================================
PHYSICS LOSS DEFINITIONS:
================================================================================
1. Archard Wear Loss (L_Archard):
   Measures violation of Archard's wear equation
   dV/dt = K·(F·v)/H

2. Thermal Expansion Loss (L_thermal):
   Measures violation of thermal expansion
   ΔL = α·L₀·ΔT

3. Force Balance Loss (L_force):
   Measures violation of cutting force balance
   F_resultant = K_c · MRR

================================================================================
Generate all training process documentation"""
    print("="*80)
    print("TRAINING FLOWCHART & PSEUDOCODE GENERATION")
    print("="*80)

    create_training_flowchart()

    generate_pseudocode()
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print("\n✅ Generated files:")
    print("   • Flowchart: results/figures_publication/figure2_training_flowchart.png")
    print("   • Pseudocode (LaTeX): results/figures_publication/pseudocode_spinn.tex")
    print("   • Pseudocode (Text): results/figures_publication/pseudocode_spinn.txt")
    print("\n📋 Usage in paper:")
    print("   • Include flowchart as Figure 2 in Methodology section")
    print("   • Reference: 'Figure 2 illustrates the complete SPINN training pipeline'")
    print("   • Include pseudocode in supplementary material or as Algorithm 1")
    print("   • Addresses Reviewer 1 request for training process clarification")
    print("\n✅ Complete!")

if __name__ == "__main__":
    main()
