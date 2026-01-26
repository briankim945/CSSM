# CSSM Model Variants

This document describes the available CSSM (Cepstral State Space Model) variants.

## Overview

All CSSM variants share a common design:
- **FFT-based spatial processing**: O(N log N) spatial mixing via spectral convolution
- **Temporal recurrence**: Associative scan for O(log T) parallel temporal processing
- **GOOM numerics**: Log-space computation for numerical stability

## Available Variants

### HGRUBilinearCSSM (`--cssm hgru_bi`) - **Recommended**

The primary CSSM variant, inspired by hGRU (horizontal Gated Recurrent Unit).

**State dimensions**: 3x3 (X, Y, Z)
- X = Excitatory state (receives input, inhibited by Y and Z)
- Y = Inhibitory state (excited by X and Z)
- Z = Interaction channel (learns to track X-Y correlation)

**Dynamics**:
```
X_t = decay_x·X - μ_I·K_I·Y - α_I·K_I·Z + U_X
Y_t = μ_E·K_E·X + decay_y·Y + α_E·K_E·Z + U_Y
Z_t = γ·X + δ·Y + ε·Z + U_Z
```

**Key features**:
- Excitation/inhibition dynamics with spatial kernels K_E, K_I
- Z channel approximates bilinear X*Y interaction while maintaining associativity
- Input-dependent gates for all coupling strengths

**Additional options**:
- `--readout_state`: Which state(s) to use for output (`xyz`, `x`, `y`, `z`, `xy`, `xz`, `yz`)
- `--pre_output_act`: Activation before output projection (`none`, `gelu`, `silu`)

### KQVCoupledCSSM (`--cssm kqv_coupled`) - **Improved KQV**

Enhanced KQV variant with cross-state coupling and recurrent gating.

**State dimensions**: 3x3 (K, Q, V) with full coupling

**Transition matrix**:
```
[decay_K·K_K    0            β_K·K_V ]   K ← V feedback
[0             decay_Q·K_Q   β_Q·K_V ]   Q ← V feedback
[γ_K·K_K       γ_Q·K_Q      decay_V·K_V]   V ← K,Q recurrent gating
```

**Key improvements over KQVCSSM**:
1. **Cross-state coupling (V→K, V→Q)**: β gates let V influence K and Q evolution
2. **Recurrent gating (K→V, Q→V)**: γ gates make V's transition depend on K,Q histories
3. **Rich output**: Uses all three states [K, Q, V] concatenated (like HGRUBilinearCSSM)

**Why this works better**: The γ terms create "recurrent gating" where V's state evolution
depends on K and Q through the transition matrix, not just input gating. This is analogous
to how Z in HGRUBilinearCSSM tracks X-Y interaction.

**Additional options**:
- `--readout_state`: Which state(s) for output (`kqv`, `k`, `q`, `v`, `kv`, `qv`)
- `--pre_output_act`: Activation before output projection (`none`, `gelu`, `silu`)

### KQVCSSM (`--cssm kqv`)

Original KQV variant with block-diagonal transitions and K*Q input gating only.

**State dimensions**: 3x3 (K, Q, V) - block diagonal (no cross-state coupling)
- K = Key state (evolves independently)
- Q = Query state (evolves independently)
- V = Value state (receives input gated by K*Q)

**Limitation**: K and Q evolve blindly - they don't receive feedback from V, so they
can't adapt to what V needs. Only V is used for output. Consider using `kqv_coupled` instead.

### GatedCSSM (`--cssm gated`)

Mamba-style gated CSSM with input-dependent integration.

**State dimensions**: Scalar per channel
- Simple h_t = A_bar * h_{t-1} + B_bar * u_t recurrence
- Input-dependent decay (Δ gate)
- Input/output gating (B, C gates)

**When to use**: Simpler baseline, fewer parameters, good for ablation studies.

## Channel Mixing Modes

All variants support:
- `--mixing depthwise` (default): Each channel independent
- `--mixing dense`: Channels can mix within blocks (LMME)
- `--block_size N`: Block size for channel mixing (1=depthwise, >1=block mixing)

## Hyperparameter Recommendations

| Task | CSSM | kernel_size | embed_dim | depth | seq_len |
|------|------|-------------|-----------|-------|---------|
| Pathfinder | hgru_bi | 11 | 32 | 1 | 8 |
| cABC | hgru_bi | 11 | 32 | 1 | 8 |
| ImageNet | hgru_bi | 11 | 384 | 12 | 8 |

## Architecture Selection

| Architecture | Use Case |
|-------------|----------|
| `--arch simple` | Pathfinder, cABC (minimal, fast) |
| `--arch vit` | ImageNet (full CSSM-ViT) |
| `--arch baseline` | ViT with attention (comparison) |
