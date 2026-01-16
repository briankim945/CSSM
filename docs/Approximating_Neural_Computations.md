# Approximating Non-Scannable Neural Computations

## Overview

This document explores how to approximate the neural computations that cannot be directly implemented in a single associative scan. The key insight is that **between scan layers**, we can perform arbitrary operations—so stacking scans with inter-layer nonlinearities dramatically expands expressivity.

---

## The Fundamental Trick: Inter-Layer Operations

**Within a scan:** Limited to linear operations on state
**Between scans:** Arbitrary nonlinear operations on layer outputs

```
[Scan₁] → Nonlinearity → Normalization → Gating → [Scan₂] → ...
          ↑_____________ ARBITRARY! _____________↑
```

This means ANY computation can be approximated by:
1. Breaking it into "linear temporal" and "nonlinear pointwise" components
2. Interleaving scans (temporal) with nonlinearities (pointwise)

---

## Approximation Strategies for Each Computation

### 1. Divisive Normalization

**Target:**
```
y = x / (σ² + Σᵢ wᵢ·xᵢ²)
```

**Strategy A: LayerNorm Between Layers**
```python
x = Scan(x)           # Temporal processing
x = LayerNorm(x)      # Divisive normalization!
# LayerNorm IS divisive normalization:
# y = (x - μ) / σ = x / σ - μ/σ
```

**Strategy B: Track Denominator as State**
```python
# In GOOM log-space, we can track the sum of squares!
# State: [X, S] where S = Σ x²

# S_t = S_{t-1} + x_t²
# In log-space: log(S_t) = log_add_exp(log(S_{t-1}), 2·log(x_t))
# This IS associative!

# Then post-scan: y = exp(log(X) - 0.5·log(S))
```

**Strategy C: Learned Soft Normalization**
```python
# Train gates to approximate normalization effect
magnitude = Dense(ctx)  # Estimate input magnitude
scale = 1.0 / (1.0 + magnitude)  # Soft normalization
x = x * scale
```

**Minimum Architecture:**
```
Scan → LayerNorm
```
**Layers needed:** 1 scan + normalization

---

### 2. Thresholding / Rectification

**Target:**
```
y = max(0, x)  or  y = σ(x)
```

**Strategy A: Post-Scan Nonlinearity**
```python
x = Scan(x)      # Linear temporal
x = GELU(x)      # Nonlinear activation
```

**Strategy B: Input Gating (Soft Threshold)**
```python
# Within scan: input gate acts as soft threshold
gate = sigmoid(Dense(ctx))  # Near 0 for weak inputs
x = x * gate                 # Weak signals suppressed
```

**Strategy C: Magnitude-Dependent Decay**
```python
# Strong signals: high decay (persist)
# Weak signals: low decay (forget quickly)
decay = sigmoid(Dense(|x|))  # Magnitude-dependent
# Approximates thresholding over time
```

**Minimum Architecture:**
```
Scan → GELU
```
**Layers needed:** 1 scan + nonlinearity

---

### 3. Multiplicative Gating / X*Y (State × State)

**Target:**
```
y = x * g(state)  where g depends on state, not just input
```

**Strategy A: Z Interaction Channel (hgru_bi)**
```python
# State: [X, Y, Z] where Z learns to track X-Y interaction
# Z receives: γ·X + δ·Y
# Z feeds back: α·K·Z (approximates α·K·(X·Y))
```

**Strategy B: Multi-Layer Multiplication**
```python
# Layer 1: Compute X features
X_out = Scan_X(input)

# Layer 2: Compute Y features
Y_out = Scan_Y(input)

# Between layers: MULTIPLY!
XY = X_out * Y_out  # This is allowed between layers!

# Layer 3: Process the product
output = Scan_XY(XY)
```

**Strategy C: Polynomial State Space (Carleman)**
```python
# Track X, Y, X², XY, Y² as extended state
# 5-dim state instead of 2-dim
# XY is directly available as state component
# BUT: requires truncating higher-order terms
```

**Strategy D: Gated Linear Unit (GLU)**
```python
# Between layers:
x = x * sigmoid(Linear(x))  # GLU-style gating
# The sigmoid(Linear(x)) acts like state-dependent gate
```

**Minimum Architecture:**
```
Scan_X ─┐
        ├─→ Multiply → Scan_XY
Scan_Y ─┘
```
**Layers needed:** 2-3 scans + multiplication

**Or with hgru_bi:**
```
Scan_3x3 (single layer, Z approximates X*Y)
```
**Layers needed:** 1 scan (but 3 channels)

---

### 4. Winner-Take-All / Competition

**Target:**
```
y_i = 1 if x_i = max(x) else 0  (hard WTA)
y_i = exp(x_i) / Σ exp(x_j)     (soft WTA / softmax)
```

**Strategy A: Strong Mutual Inhibition**
```python
# Large negative off-diagonal in transition matrix
# [decay    -strong]
# [-strong   decay ]
# Creates competition, but soft, not hard WTA
```

**Strategy B: Post-Scan Softmax**
```python
x = Scan(x)        # Process with inhibition
x = Softmax(x)     # Hard competition
```

**Strategy C: Iterative Refinement**
```python
for i in range(K):
    x = Scan(x)                    # Mutual inhibition
    x = Softmax(x, temperature=T)  # Competition
    T = T * 0.9                    # Anneal temperature
# Temperature annealing → hard WTA
```

**Strategy D: Top-K Selection Between Layers**
```python
x = Scan(x)
x = top_k_mask(x, k=1) * x  # Zero out all but winner
x = Scan(x)                  # Continue with winner
```

**Minimum Architecture:**
```
Scan → Softmax
```
**Layers needed:** 1 scan + softmax

**For hard WTA:**
```
[Scan → Softmax(temp↓)] × K
```
**Layers needed:** K iterations with annealing

---

### 5. Coincidence Detection

**Target:**
```
y = f(x₁ · x₂)  # Detect when both x₁ and x₂ are active
```

**Strategy A: Z Channel**
```python
# hgru_bi: Z = γ·X + δ·Y
# High Z indicates both X and Y are active
# Not exact product, but correlation proxy
```

**Strategy B: Multi-Layer Product**
```python
X_features = Scan_X(input)
Y_features = Scan_Y(input)
coincidence = X_features * Y_features  # Between layers!
output = Scan(coincidence)
```

**Strategy C: Sum + Threshold**
```python
# If X, Y ∈ [0,1], then X + Y > 1.5 implies both > 0.75
x = Scan([X, Y])
coincidence = (X + Y > threshold)  # Between layers
```

**Strategy D: Learned Coincidence via MLP**
```python
x = Scan(x)
coincidence = MLP(x)  # Learn to detect coincidences
# MLP can learn: coincidence ≈ X * Y
```

**Minimum Architecture:**
```
Scan → MLP (learns X*Y pattern)
```
**Layers needed:** 1 scan + MLP

---

### 6. Hebbian Learning (X*Y Weight Updates)

**Target:**
```
Δw_ij = η · xᵢ · xⱼ
```

**Key insight:** This is about **learning**, not forward computation.

**Strategy A: Outer Loop**
```python
# Forward pass: normal scan
output = Scan(input)

# Backward pass: gradients compute Hebbian-like updates
# ∂L/∂w includes terms like x_pre * x_post
# Backprop naturally computes correlation-based updates!
```

**Strategy B: Fast Weights / Hypernetwork**
```python
# Compute "fast weights" from input correlations
correlation = Scan_correlation(input)  # Track X*Y over time
fast_weights = Linear(correlation)

# Use fast weights to modulate main scan
output = Scan(input, weights=base_weights + fast_weights)
```

**Strategy C: Attention as Hebbian**
```python
# Attention IS a form of Hebbian computation!
# attn(Q, K) = softmax(Q @ K.T) computes correlations
# Could add attention between scan layers
```

**Minimum Architecture:**
```
Scan₁ → Compute Correlation → Scan₂(modulated)
```
**Layers needed:** 2 scans + correlation computation

---

## Sufficient Architectures

### The "Scan Transformer" Block

A single block that can approximate all neural computations:

```python
class ScanTransformerBlock(nn.Module):
    """
    Scan + MLP block analogous to Transformer's Attention + MLP.

    Within scan: Linear temporal mixing
    MLP: Nonlinearity, implicit normalization, gating
    LayerNorm: Explicit normalization
    """
    def __call__(self, x):
        # ===== Temporal Mixing (like Attention) =====
        residual = x
        x = LayerNorm(x)
        x = Scan(x)           # O(log T) parallel
        x = residual + x      # Residual connection

        # ===== Channel Mixing (like FFN) =====
        residual = x
        x = LayerNorm(x)
        x = Linear(x)         # Up-project
        x = GELU(x)           # Nonlinearity (thresholding)
        x = Linear(x)         # Down-project (can implement gating)
        x = residual + x

        return x
```

### What This Block Provides

| Computation | How |
|-------------|-----|
| Linear Filtering | Scan (FFT convolution) |
| Divisive Normalization | LayerNorm |
| Thresholding | GELU in MLP |
| Input-Dependent Gating | Scan gates + MLP |
| Temporal Integration | Scan accumulation |
| Adaptation | Opponent channels in Scan |
| Working Memory | High decay in Scan |
| Oscillations | Complex eigenvalues in Scan |
| Center-Surround | K_E - K_I kernels |
| Feedback Amplification | Off-diagonal Scan terms |

### What Requires Multiple Blocks

| Computation | Architecture |
|-------------|--------------|
| X*Y Multiplication | Block₁ → Multiply → Block₂ |
| Winner-Take-All | Block₁ → Softmax → Block₂ |
| Coincidence Detection | Block₁(X) * Block₁(Y) → Block₂ |
| Sequence Detection | Deep stack for complex patterns |
| Hebbian (fast weights) | Block₁ → Correlation → Block₂(modulated) |

---

## Depth Requirements for Universal Approximation

### Theoretical Results

**Claim 1:** A single Scan layer followed by a 2-layer MLP can approximate any continuous function of a finite window of the input.

*Proof sketch:*
- Scan computes weighted sum of past inputs (linear temporal filter)
- MLP is a universal function approximator (Cybenko, 1989)
- Composition of linear temporal + nonlinear pointwise = universal

**Claim 2:** A stack of N Scan→MLP blocks can approximate any continuous function of the full input sequence with error O(1/N).

*Proof sketch:*
- Each block refines the representation
- Residual connections enable gradient flow
- Similar to ResNet universal approximation results

### Practical Depth Guidelines

| Target Computation | Minimum Depth | Notes |
|-------------------|---------------|-------|
| Linear filtering | 1 | Single scan |
| Normalization | 1 | Scan + LayerNorm |
| Thresholding | 1 | Scan + GELU |
| X*Y (exact) | 2-3 | Separate branches + multiply |
| X*Y (approximate) | 1 | hgru_bi with Z channel |
| WTA (soft) | 1 | Scan + Softmax |
| WTA (hard) | 2-3 | Iterative refinement |
| Coincidence | 2 | Like X*Y |
| Complex sequences | 4-6 | Need compositional depth |
| Full cortical circuit | 6-12 | Empirical from vision models |

---

## The Complete Architecture

For maximum expressivity, use:

```python
class CompleteScanNetwork(nn.Module):
    """
    Full architecture approximating all neural computations.
    """
    num_layers: int = 6

    def __call__(self, x):
        # Stem: Input projection
        x = PatchEmbed(x)

        # Main body: Stack of ScanTransformer blocks
        for i in range(self.num_layers):
            x = ScanTransformerBlock(x)

            # Optional: Inter-block operations for specific computations
            if i == self.num_layers // 2:
                # Mid-network WTA / competition
                x = Softmax(x, axis=-1)

            if i == self.num_layers // 3:
                # Branch and multiply for X*Y
                x_branch = Linear(x)
                x = x * sigmoid(x_branch)  # GLU-style

        # Head: Output projection
        x = LayerNorm(x)
        x = Linear(x)

        return x
```

---

## Comparison: Depth vs Width vs Channels

| Strategy | Pros | Cons |
|----------|------|------|
| **More depth** | More compositions, complex functions | Slower, gradient issues |
| **More width** | Richer representations | Memory cost |
| **More channels (2→3)** | Z approximates X*Y in single layer | 50% more compute per layer |
| **Chunked scan** | True nonlinearity within sequence | O(chunks × log(chunk_size)) |

### Recommended Trade-offs

**For speed:** Use hgru_bi (3 channels) with moderate depth (4-6 layers)
**For accuracy:** Use hgru (2 channels) with more depth (8-12 layers)
**For very long sequences:** Use chunked scan with nonlinearity between chunks

---

## Summary Table: How to Approximate Each Computation

| Computation | Single-Layer Approx | Multi-Layer Exact | Min Layers |
|-------------|--------------------|--------------------|------------|
| Divisive Norm | LayerNorm | LayerNorm | 1 |
| Thresholding | Soft gates | GELU between | 1 |
| X*Y (bilinear) | Z channel | Branch→Multiply→Merge | 1-3 |
| WTA | Strong inhibition | Softmax between | 1-2 |
| Coincidence | Z channel | Product between layers | 1-2 |
| Hebbian | N/A | Outer loop | N/A |
| Sequence detection | State accumulation | Deep stack | 4+ |

---

## Key Takeaways

1. **Single scan + post-processing** handles: normalization, thresholding, soft competition

2. **Multiple scans with inter-layer ops** handles: multiplication, hard WTA, coincidence

3. **The Z channel (hgru_bi)** is a single-layer approximation to X*Y that avoids the depth cost

4. **Universal approximation** is achievable with enough depth, but some computations are "closer" to linear and need fewer layers

5. **The Scan Transformer block** (Scan + LayerNorm + MLP) is a sufficient building block for approximating any neural computation

6. **Hebbian learning** fundamentally requires the outer training loop—it's about weight updates, not forward computation

---

## Open Questions

1. **Optimal depth-width trade-off:** For a fixed compute budget, is it better to use deeper 2-channel or shallower 3-channel?

2. **Learned inter-layer operations:** Can we learn WHEN to apply softmax, multiply, etc. rather than fixing it?

3. **Continuous-depth scans:** Neural ODEs with scan-style operators?

4. **Biological plausibility:** The brain doesn't have clean "layers"—how do continuous recurrent dynamics relate to stacked scans?

5. **Efficient X*Y:** Is there a better approximation than Z channel that's still O(log T)?
