import torch
import numpy as np
import matplotlib.pyplot as plt
import os

diagnostic_dir = 'attn_diagnostics'
files = sorted([f for f in os.listdir(diagnostic_dir) if f.endswith('.pt')])
print(f"Found {len(files)} saved attention matrices")

# ── Load all saved matrices ───────────────────────────────────────────────────
# Each file is [B, H, N, N]
# calls 1,4,7,... = layer 1 (every 3rd starting from 1)
# calls 2,5,8,... = layer 2
# calls 3,6,9,... = layer 3

all_sv = {1: [], 2: [], 3: []}  # layer -> list of singular values

for i, fname in enumerate(files):
    layer = (i % 3) + 1  # 1, 2, or 3
    A = torch.load(os.path.join(diagnostic_dir, fname))
    B, H, N, _ = A.shape

    for b in range(B):
        for h in range(H):
            mat = A[b, h].numpy()
            _, s, _ = np.linalg.svd(mat)
            all_sv[layer].append(s)

# ── Per-layer analysis ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for layer_idx, layer in enumerate([1, 2, 3]):
    sv = np.array(all_sv[layer])       # [num_samples, N]
    sv_mean = sv.mean(axis=0)
    sv_std  = sv.std(axis=0)
    cumulative = np.cumsum(sv_mean) / sv_mean.sum()
    N = sv.shape[1]

    print(f"\n── Layer {layer} ──────────────────────────")
    print(f"  Samples analysed: {len(sv)}")
    for i, (s, c) in enumerate(zip(sv_mean, cumulative)):
        print(f"  rank {i+1:2d}: sv = {s:.4f}   cumulative = {c:.1%}")
    print(f"  Rank for 80%: {np.searchsorted(cumulative, 0.80) + 1}")
    print(f"  Rank for 90%: {np.searchsorted(cumulative, 0.90) + 1}")
    print(f"  Rank for 95%: {np.searchsorted(cumulative, 0.95) + 1}")
    print(f"  Rank for 99%: {np.searchsorted(cumulative, 0.99) + 1}")

    # Top row: singular value decay per layer
    ax = axes[0, layer_idx]
    ax.plot(range(1, N+1), sv_mean, 'o-', color='steelblue', linewidth=2)
    ax.fill_between(range(1, N+1), sv_mean - sv_std, sv_mean + sv_std,
                    alpha=0.2, color='steelblue')
    ax.set_title(f'Layer {layer} — singular value decay')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Singular value')
    ax.grid(True, alpha=0.3)

    # Bottom row: cumulative variance per layer
    ax = axes[1, layer_idx]
    ax.plot(range(1, N+1), cumulative * 100, 's-', color='coral', linewidth=2)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='90%')
    ax.axhline(y=95, color='gray', linestyle=':',  alpha=0.7, label='95%')
    ax.set_title(f'Layer {layer} — cumulative variance')
    ax.set_xlabel('Rank k')
    ax.set_ylabel('Cumulative variance (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('SVD analysis of variate attention matrix — all 3 encoder layers', 
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('svd_all_layers.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: svd_all_layers.png")