"""
Mathematical primitives for CSSM (Cepstral State Space Models).

Implements associative scan operators for log-semiring computations
using proper GOOM (Generalized Order of Magnitude) primitives.

Supports three modes:
1. Scalar (depthwise): Each channel independent - O(C) complexity
2. 2x2 Matrix (opponent): X/Y coupled oscillator - O(C) complexity
3. Block-diagonal (dense mixing): LMME channel mixing - O(C × block_size²) complexity
"""

import jax.numpy as jnp
from typing import Tuple

from .operations import (
    log_add_exp, log_matmul_2x2, log_matvec_2x2,
    log_matmul_3x3, log_matvec_3x3,
    log_matmul_block, log_matvec_block
)


def cssm_scalar_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for Standard CSSM (Scalar/Diagonal).

    Computes: (k_j, u_j) o (k_i, u_i) = (k_j + k_i, LSE(k_j + u_i, u_j))

    This implements the recurrence h_t = k * h_{t-1} + u_t in log-space,
    where multiplication becomes addition and addition becomes log-sum-exp.

    All values are expected to be in GOOM representation (complex log-space).

    Args:
        carry_i: Tuple of (kernel_log, input_log) for position i
        carry_j: Tuple of (kernel_log, input_log) for position j

    Returns:
        Combined carry for the associative scan
    """
    k_i, u_i = carry_i
    k_j, u_j = carry_j

    # In log-space: multiplication -> addition
    k_new = k_j + k_i

    # In log-space: addition -> log-sum-exp (using GOOM)
    u_new = log_add_exp(k_j + u_i, u_j)

    return k_new, u_new


def cssm_matrix_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for 2x2 Linear Opponent CSSM (hgru mode).

    ============================================================================
    WHAT THIS COMPUTES (Linear Recurrence)
    ============================================================================

    Original dynamics in LINEAR space:
        [X_t]   [decay_x    -μ_I·K_I] [X_{t-1}]   [U_X]
        [Y_t] = [μ_E·K_E     decay_y] [Y_{t-1}] + [U_Y]

    Or written out:
        X_t = decay_x·X_{t-1} - μ_I·K_I·Y_{t-1} + U_X   (Y inhibits X via K_I)
        Y_t = μ_E·K_E·X_{t-1} + decay_y·Y_{t-1} + U_Y   (X excites Y via K_E)

    The K_I and K_E are FFT'd convolution kernels, so in spectral domain
    the matrix multiplication IS the convolution!

    ============================================================================
    DOMAIN TRANSFORMATIONS
    ============================================================================

    SPATIAL         SPECTRAL (FFT)       LOG-SPECTRAL (GOOM)
    -------         --------------       -------------------
    conv(A,B)  -->  A_hat * B_hat   -->  log(A) + log(B)
    a + b      -->  a + b           -->  log_add_exp(log(a), log(b))
    a * b      -->  a * b           -->  log(a) + log(b)

    So in GOOM:
    - Matrix multiplication → log_matmul (uses log_add_exp internally)
    - Matrix-vector mult → log_matvec (uses log_add_exp internally)
    - Addition → log_add_exp

    ============================================================================
    ASSOCIATIVE SCAN SEMANTICS
    ============================================================================

    The scan combines results from position i (earlier) and j (later).

    carry_i = (K_i, u_i) represents accumulated computation from positions 0..i
    carry_j = (K_j, u_j) represents accumulated computation from positions i+1..j

    To combine them:
    1. K_new = K_j @ K_i  (compose transition matrices)
       - In log-space: log_matmul_2x2
       - This accumulates the "decay and coupling" over time
       - Receptive field GROWS as K's accumulate (conv of conv of conv...)

    2. u_new = K_j @ u_i + u_j  (propagate inputs through transitions)
       - K_j @ u_i: Apply later transitions to earlier accumulated inputs
       - + u_j: Add the later inputs
       - In log-space: log_matvec then log_add_exp

    ============================================================================
    STEP-BY-STEP EXAMPLE
    ============================================================================

    Say we have 4 timesteps: t=0,1,2,3

    Sequential would compute:
        state_1 = K_1 @ state_0 + u_1
        state_2 = K_2 @ state_1 + u_2
        state_3 = K_3 @ state_2 + u_3

    Parallel scan does:
        Level 0: (K_0,u_0), (K_1,u_1), (K_2,u_2), (K_3,u_3)

        Level 1: Combine pairs
            (K_0,u_0) ⊕ (K_1,u_1) = (K_1@K_0, K_1@u_0 + u_1)
            (K_2,u_2) ⊕ (K_3,u_3) = (K_3@K_2, K_3@u_2 + u_3)

        Level 2: Combine results
            Result gives us state_3 = K_3@K_2@K_1@(K_0@init + u_0) + K_3@K_2@u_1 + K_3@u_2 + u_3

    This is O(log T) depth instead of O(T)!

    ============================================================================

    Args:
        carry_i: Tuple of (K_matrix_log, u_vector_log) for earlier positions
                 K has shape (..., 2, 2), u has shape (..., 2)
        carry_j: Tuple of (K_matrix_log, u_vector_log) for later positions

    Returns:
        Combined carry: (K_new, u_new)
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j

    # =========================================================================
    # STEP 1: Compose transition matrices
    # =========================================================================
    # K_new = K_j @ K_i in log-space
    #
    # This computes the combined transition from positions 0..j
    # In log-space, matrix multiply uses log_add_exp for the inner sums:
    #   C[i,j] = log_add_exp(A[i,0] + B[0,j], A[i,1] + B[1,j])
    #          = log(exp(A[i,0] + B[0,j]) + exp(A[i,1] + B[1,j]))
    #          = log(A[i,0]*B[0,j] + A[i,1]*B[1,j])  in linear space
    #
    # The spatial convolution kernels K_I, K_E are baked into K.
    # Accumulating K matrices = accumulating convolutions = growing receptive field!
    K_new = log_matmul_2x2(K_j, K_i)

    # =========================================================================
    # STEP 2: Propagate accumulated inputs through later transitions
    # =========================================================================
    # Ku_i = K_j @ u_i in log-space
    #
    # u_i contains accumulated weighted inputs from positions 0..i
    # K_j contains the transition matrix from position j
    # This applies the later transition to the earlier accumulated state
    #
    # In log-space: y[k] = log_add_exp(K[k,0] + u[0], K[k,1] + u[1])
    # Which is: y[k] = log(K[k,0]*u[0] + K[k,1]*u[1]) in linear space
    Ku_i = log_matvec_2x2(K_j, u_i)

    # =========================================================================
    # STEP 3: Add later inputs
    # =========================================================================
    # u_new = Ku_i + u_j in log-space (element-wise)
    #
    # log_add_exp computes log(exp(a) + exp(b)) = log(a_linear + b_linear)
    # This adds the propagated earlier inputs to the later inputs
    #
    # u_new[0] = total accumulated X input
    # u_new[1] = total accumulated Y input
    u_new_0 = log_add_exp(Ku_i[..., 0], u_j[..., 0])
    u_new_1 = log_add_exp(Ku_i[..., 1], u_j[..., 1])
    u_new = jnp.stack([u_new_0, u_new_1], axis=-1)

    return K_new, u_new


def cssm_block_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray],
    block_size: int = 32
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for Block-Diagonal CSSM (LMME Channel Mixing).

    Implements LMME (Log-Matrix-Matrix-Exp) for channel mixing following
    the CSSM paper. Channels are grouped into blocks that mix internally.

    This enables channel mixing with O(C × block_size²) complexity instead
    of O(C³) for full mixing or O(C) for depthwise.

    The LMME operation:
        LMME(A, B)_ij = LSE_k(A_ik + B_kj)

    Args:
        carry_i: Tuple of (K_block_log, u_block_log) for position i
                 K has shape (..., num_blocks, block_size, block_size)
                 u has shape (..., num_blocks, block_size)
        carry_j: Tuple of (K_block_log, u_block_log) for position j
        block_size: Size of each channel mixing block

    Returns:
        Combined carry: (K_new, u_new) where
            K_new = LMME(K_j, K_i) (log-matrix multiplication per block)
            u_new = LMME(K_j, u_i) + u_j (log-matrix-vector + log-sum-exp)
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j

    # K_new = K_j @ K_i in log-space (per block)
    K_new = log_matmul_block(K_j, K_i, block_size)

    # u_new = K_j @ u_i + u_j in log-space
    # First: block matrix-vector multiplication
    Ku_i = log_matvec_block(K_j, u_i, block_size)

    # Then: element-wise log-sum-exp with u_j
    u_new = log_add_exp(Ku_i, u_j)

    return K_new, u_new


def make_block_scan_op(block_size: int):
    """
    Create a block scan operator with fixed block_size.

    JAX's associative_scan requires a binary operator, but cssm_block_scan_op
    has an extra block_size parameter. This factory creates a closure with
    the block_size baked in.

    Args:
        block_size: Size of each channel mixing block

    Returns:
        Binary scan operator suitable for jax.lax.associative_scan
    """
    def scan_op(carry_i, carry_j):
        return cssm_block_scan_op(carry_i, carry_j, block_size)
    return scan_op


# =============================================================================
# 3x3 CSSM with Interaction Channel (hgru_bi mode)
# =============================================================================
# State: [X, Y, Z] where:
#   X = Excitatory state (receives input, inhibited by Y and Z)
#   Y = Inhibitory state (excited by X)
#   Z = Interaction channel (learns to track X-Y interaction for "bilinear-like" effect)
#
# Key insight: Z can learn to approximate X*Y interaction through linear dynamics!
# The 3x3 matrix lets Z mix X and Y contributions, then feed back into X and Y.

def cssm_3x3_matrix_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for 3x3 CSSM with Interaction Channel (hgru_bi mode).

    ============================================================================
    WHAT THIS COMPUTES (3-State Linear Recurrence with Interaction)
    ============================================================================

    Original dynamics in LINEAR space:
        [X_t]   [decay_x    -μ_I·K_I    -α_I·K_I] [X_{t-1}]   [U_X]
        [Y_t] = [μ_E·K_E     decay_y    +α_E·K_E] [Y_{t-1}] + [U_Y]
        [Z_t]   [γ           δ           ε      ] [Z_{t-1}]   [U_Z]

    Or written out:
        X_t = decay_x·X - μ_I·K_I·Y - α_I·K_I·Z + U_X   (inhibited by Y AND Z)
        Y_t = μ_E·K_E·X + decay_y·Y + α_E·K_E·Z + U_Y   (excited by X AND Z)
        Z_t = γ·X + δ·Y + ε·Z + U_Z                     (learns X-Y interaction)

    ============================================================================
    WHY THIS APPROXIMATES BILINEAR
    ============================================================================

    True hGRU bilinear: X_t = ... - α·K_I·X_{t-1}·Y_{t-1}

    We can't do X*Y directly in associative scan (breaks associativity).
    But Z can LEARN to track an interaction:

    If Z learns: Z ≈ β·X + γ·Y + accumulated_interaction
    Then: -α_I·K_I·Z ≈ -α_I·K_I·(β·X + γ·Y + ...)

    This isn't exactly X*Y, but it's a learnable proxy that:
    1. Depends on both X and Y history
    2. Feeds back into X (inhibition) and Y (excitation)
    3. Can capture correlation/interaction patterns

    ============================================================================
    THE 3x3 TRANSITION MATRIX
    ============================================================================

    [decay_x    -μ_I·K_I    -α_I·K_I]    Row 0: X update
    [μ_E·K_E     decay_y    +α_E·K_E]    Row 1: Y update
    [γ           δ           ε      ]    Row 2: Z update (interaction tracker)

    Key positions:
    - (0,2) = -α_I·K_I: Z inhibits X (like X*Y would)
    - (1,2) = +α_E·K_E: Z excites Y (like X*Y would)
    - (2,0) = γ: X contributes to Z
    - (2,1) = δ: Y contributes to Z
    - (2,2) = ε: Z self-decay

    ============================================================================
    SCAN MECHANICS (same as 2x2, just bigger matrices)
    ============================================================================

    carry_i = (K_i, u_i): Accumulated from positions 0..i
    carry_j = (K_j, u_j): Accumulated from positions i+1..j

    Combine:
    1. K_new = K_j @ K_i  (3x3 log-matmul)
    2. u_new = K_j @ u_i + u_j  (3x3 log-matvec + log-add-exp)

    ============================================================================

    Args:
        carry_i: Tuple of (K_matrix_log, u_vector_log) for earlier positions
                 K has shape (..., 3, 3), u has shape (..., 3)
        carry_j: Tuple of (K_matrix_log, u_vector_log) for later positions

    Returns:
        Combined carry: (K_new, u_new)
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j

    # =========================================================================
    # STEP 1: Compose 3x3 transition matrices
    # =========================================================================
    # K_new = K_j @ K_i in log-space
    #
    # For 3x3: C[i,j] = log_add_exp(A[i,0]+B[0,j], A[i,1]+B[1,j], A[i,2]+B[2,j])
    #
    # This accumulates transitions including the Z interaction channel.
    # Over time, Z's contribution to X and Y grows through the off-diagonal terms.
    K_new = log_matmul_3x3(K_j, K_i)

    # =========================================================================
    # STEP 2: Propagate accumulated inputs through later transitions
    # =========================================================================
    # Ku_i = K_j @ u_i in log-space (3x3 matrix × 3-vector)
    #
    # For each output k: y[k] = log_add_exp(K[k,0]+u[0], K[k,1]+u[1], K[k,2]+u[2])
    #
    # This mixes X, Y, Z contributions according to the transition matrix
    Ku_i = log_matvec_3x3(K_j, u_i)

    # =========================================================================
    # STEP 3: Add later inputs
    # =========================================================================
    # u_new = Ku_i + u_j in log-space (element-wise log-sum-exp)
    #
    # u_new[0] = accumulated X state
    # u_new[1] = accumulated Y state
    # u_new[2] = accumulated Z state (interaction memory)
    u_new_0 = log_add_exp(Ku_i[..., 0], u_j[..., 0])
    u_new_1 = log_add_exp(Ku_i[..., 1], u_j[..., 1])
    u_new_2 = log_add_exp(Ku_i[..., 2], u_j[..., 2])
    u_new = jnp.stack([u_new_0, u_new_1, u_new_2], axis=-1)

    return K_new, u_new


def cssm_3x3_bilinear_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray],
    K_bilinear: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for 3x3 CSSM with bilinear X*Z term.

    Computes: X_t = linear_terms - K_I*ν*X*Z (hGRU-style X² inhibition)

    The bilinear term log(X*Z) = log(X) + log(Z) is computed in log-space
    and added to the X update.

    State vector: [X, Y, Z] where Z = X_{t-1}

    Args:
        carry_i: Tuple of (K_matrix_log, u_vector_log) for position i
                 K has shape (..., 3, 3), u has shape (..., 3)
        carry_j: Tuple of (K_matrix_log, u_vector_log) for position j
        K_bilinear: Bilinear coefficient log(K_I * ν) with phase π for subtraction
                    Shape: (...) matching spatial dimensions

    Returns:
        Combined carry: (K_new, u_new) where u_new[0] includes bilinear term
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j

    # Standard 3x3 matrix composition
    K_new = log_matmul_3x3(K_j, K_i)
    Ku_i = log_matvec_3x3(K_j, u_i)

    # === BILINEAR TERM: X * Z ===
    # In log-space: log(X * Z) = log(X) + log(Z)
    log_X = u_i[..., 0]  # log(X_{t-1})
    log_Z = u_i[..., 2]  # log(Z_{t-1}) = log(X_{t-2})
    log_XZ = log_X + log_Z  # log(X * Z) - product becomes sum!

    # Bilinear inhibition: -K_I * ν * X * Z
    # K_bilinear includes phase π for the negative sign
    bilinear_inhib = K_bilinear + log_XZ

    # Combine: X_new = LSE(linear_terms, bilinear_term)
    u_new_X = log_add_exp(
        log_add_exp(Ku_i[..., 0], u_j[..., 0]),
        bilinear_inhib
    )
    u_new_Y = log_add_exp(Ku_i[..., 1], u_j[..., 1])
    u_new_Z = log_add_exp(Ku_i[..., 2], u_j[..., 2])

    u_new = jnp.stack([u_new_X, u_new_Y, u_new_Z], axis=-1)
    return K_new, u_new


def make_3x3_bilinear_scan_op(K_bilinear: jnp.ndarray):
    """
    Create a 3x3 bilinear scan operator with fixed K_bilinear.

    JAX's associative_scan requires a binary operator, but cssm_3x3_bilinear_scan_op
    has an extra K_bilinear parameter. This factory creates a closure with
    K_bilinear baked in.

    Args:
        K_bilinear: Bilinear coefficient log(K_I * ν) in GOOM representation
                    Should include phase π for subtraction (negative inhibition)

    Returns:
        Binary scan operator suitable for jax.lax.associative_scan
    """
    def scan_op(carry_i, carry_j):
        return cssm_3x3_bilinear_scan_op(carry_i, carry_j, K_bilinear)
    return scan_op


def cssm_hgru_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray],
    K_inhib_bilinear: jnp.ndarray,
    K_excit_bilinear: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for hGRU-style CSSM with X*Y bilinear terms.

    Implements true hGRU equations:
        X_t = linear_X - α·K_I·X·Y + U   (X*Y inhibits X)
        Y_t = linear_Y + α·K_E·X·Y       (X*Y excites Y)

    In log-space: log(X*Y) = log(X) + log(Y) — just addition!

    State vector: [X, Y] (2-element)

    Args:
        carry_i: Tuple of (K_matrix_log, u_vector_log) for position i
                 K has shape (..., 2, 2), u has shape (..., 2)
        carry_j: Tuple of (K_matrix_log, u_vector_log) for position j
        K_inhib_bilinear: log(α·K_I) with phase π for subtraction (inhibits X)
        K_excit_bilinear: log(α·K_E) with phase 0 for addition (excites Y)

    Returns:
        Combined carry: (K_new, u_new) with bilinear X*Y terms included
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j

    # Standard 2x2 matrix composition (linear part)
    K_new = log_matmul_2x2(K_j, K_i)
    Ku_i = log_matvec_2x2(K_j, u_i)

    # === BILINEAR TERM: X * Y ===
    # In log-space: log(X * Y) = log(X) + log(Y)
    log_X = u_i[..., 0]  # log(X accumulated)
    log_Y = u_i[..., 1]  # log(Y accumulated)
    # Clip to prevent unbounded accumulation over many epochs (causes NaN around epoch 40+)
    log_XY = jnp.clip(log_X + log_Y, -50.0, 50.0)  # log(X * Y) - bounded!

    # X update: add -α·K_I·X·Y (inhibition from Y, gated by X)
    # K_inhib_bilinear includes phase π for the negative sign
    bilinear_inhib = K_inhib_bilinear + log_XY
    u_new_X = log_add_exp(
        log_add_exp(Ku_i[..., 0], u_j[..., 0]),
        bilinear_inhib
    )

    # Y update: add +α·K_E·X·Y (excitation from X, gated by Y)
    # K_excit_bilinear has phase 0 for positive addition
    bilinear_excit = K_excit_bilinear + log_XY
    u_new_Y = log_add_exp(
        log_add_exp(Ku_i[..., 1], u_j[..., 1]),
        bilinear_excit
    )

    u_new = jnp.stack([u_new_X, u_new_Y], axis=-1)
    return K_new, u_new


def make_hgru_scan_op(K_inhib_bilinear: jnp.ndarray, K_excit_bilinear: jnp.ndarray):
    """
    Create an hGRU-style scan operator with fixed bilinear coefficients.

    JAX's associative_scan requires a binary operator, but cssm_hgru_scan_op
    has extra parameters. This factory creates a closure with them baked in.

    Args:
        K_inhib_bilinear: log(α·K_I) in GOOM with phase π (for X inhibition)
        K_excit_bilinear: log(α·K_E) in GOOM with phase 0 (for Y excitation)

    Returns:
        Binary scan operator suitable for jax.lax.associative_scan
    """
    def scan_op(carry_i, carry_j):
        return cssm_hgru_scan_op(carry_i, carry_j, K_inhib_bilinear, K_excit_bilinear)
    return scan_op


# =============================================================================
# KQV-CSSM: Transformer-inspired K*Q bilinear gating
# =============================================================================
# State: [K, Q, V] where:
#   K = Key state (accumulates spatial features via conv)
#   Q = Query state (accumulates spatial features via conv)
#   V = Value state (receives input GATED by K*Q product)
#
# Key insight: K*Q (Hadamard product) gates input flow to V, similar to
# how attention = softmax(Q @ K^T) @ V gates value contribution.
# The bilinear K*Q term is added to the INPUT accumulation (u), not the
# transition matrix, which preserves associativity.

def cssm_kqv_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for KQV-CSSM with K*Q bilinear gating.

    ============================================================================
    WHAT THIS COMPUTES (3-State with Bilinear Gating)
    ============================================================================

    State: [K, Q, V] where V receives input gated by K*Q.

    Dynamics in linear space:
        K_t = decay_K * conv(K_{t-1}, W_K) + B_K * U
        Q_t = decay_Q * conv(Q_{t-1}, W_Q) + B_Q * U
        V_t = decay_V * conv(V_{t-1}, W_V) + (K_{t-1} * Q_{t-1}) * B_V * U

    K and Q evolve independently via diagonal transitions.
    V receives bilinear K*Q gating on its input term.

    ============================================================================
    WHY THIS IS ASSOCIATIVE
    ============================================================================

    The bilinear term K*Q is added to the INPUT accumulation (u_j), not the
    transition matrix (K_j). Since u_i contains previously accumulated states,
    the operation:
        u_new = K_j @ u_i + u_j + bilinear_term_from_u_i
    is associative.

    ============================================================================
    LOG-SPACE COMPUTATION
    ============================================================================

    In GOOM log-space:
        log(K * Q) = log(K) + log(Q)  (no clipping - let gradients flow)

    The gated V input becomes:
        gated_input = log_K + log_Q + log_B_V + log_U_V

    ============================================================================

    Args:
        carry_i: Tuple of (K_matrix_log, u_vector_log) for earlier positions
                 K has shape (..., 3, 3), u has shape (..., 3)
        carry_j: Tuple of (K_matrix_log, u_vector_log) for later positions

    Returns:
        Combined carry: (K_new, u_new)
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j

    # =========================================================================
    # STEP 1: Compose 3x3 transition matrices (diagonal for K/Q/V)
    # =========================================================================
    K_new = log_matmul_3x3(K_j, K_i)

    # =========================================================================
    # STEP 2: Propagate accumulated inputs through later transitions
    # =========================================================================
    Ku_i = log_matvec_3x3(K_j, u_i)

    # =========================================================================
    # STEP 3: K and Q - standard linear accumulation
    # =========================================================================
    u_new_K = log_add_exp(Ku_i[..., 0], u_j[..., 0])
    u_new_Q = log_add_exp(Ku_i[..., 1], u_j[..., 1])

    # =========================================================================
    # STEP 4: BILINEAR GATING - V input gated by K*Q
    # =========================================================================
    # In log-space: log(K * Q) = log(K) + log(Q)
    log_K = u_i[..., 0]  # log(K accumulated)
    log_Q = u_i[..., 1]  # log(Q accumulated)
    log_KQ = log_K + log_Q  # K*Q in log-space (no clipping for gradient flow)

    # V accumulates: linear_V + (K*Q) * U_V
    # The K*Q term gates the input to V (u_j[..., 2])
    gated_input = log_KQ + u_j[..., 2]
    u_new_V = log_add_exp(Ku_i[..., 2], gated_input)

    u_new = jnp.stack([u_new_K, u_new_Q, u_new_V], axis=-1)
    return K_new, u_new


def cssm_kqv_block_scan_op(
    carry_i: Tuple[jnp.ndarray, jnp.ndarray],
    carry_j: Tuple[jnp.ndarray, jnp.ndarray],
    block_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Associative operator for KQV-CSSM with block-diagonal channel mixing.

    ============================================================================
    BLOCK-DIAGONAL CHANNEL MIXING (Multi-Head Analogy)
    ============================================================================

    Channels are grouped into blocks of size `block_size`:
    - num_heads = C / block_size (like attention heads)
    - Within each block, channels can mix during recurrence
    - Blocks operate independently (like parallel attention heads)

    State vector per block: [K_1, ..., K_d, Q_1, ..., Q_d, V_1, ..., V_d]
    where d = block_size.

    Transition matrix per block: (3d x 3d) with structure:
        [K_block    0          0        ]
        [0          Q_block    0        ]
        [0          0          V_block  ]

    Each *_block is (d x d), allowing channel mixing within K, Q, V separately.

    ============================================================================
    BILINEAR GATING WITH BLOCKS
    ============================================================================

    The K*Q gating is computed per-channel within each block:
        gate[c] = K[c] * Q[c]  for c in block

    This gates the corresponding V[c] input. No cross-channel gating.

    ============================================================================

    Args:
        carry_i: Tuple of (K_matrix_log, u_vector_log) for earlier positions
                 K has shape (..., num_blocks, 3*block_size, 3*block_size)
                 u has shape (..., num_blocks, 3*block_size)
        carry_j: Tuple of (K_matrix_log, u_vector_log) for later positions
        block_size: Number of channels per block (d)

    Returns:
        Combined carry: (K_new, u_new)
    """
    K_i, u_i = carry_i
    K_j, u_j = carry_j
    d = block_size

    # =========================================================================
    # STEP 1: Block matrix multiplication for transition composition
    # =========================================================================
    # K_new = K_j @ K_i per block (using log-space matmul)
    K_new = log_matmul_block(K_j, K_i, 3 * d)

    # =========================================================================
    # STEP 2: Block matrix-vector for state propagation
    # =========================================================================
    Ku_i = log_matvec_block(K_j, u_i, 3 * d)

    # =========================================================================
    # STEP 3: Standard accumulation for K and Q portions
    # =========================================================================
    # K channels: indices 0 to d-1
    # Q channels: indices d to 2d-1
    # V channels: indices 2d to 3d-1
    u_new_KQ = log_add_exp(Ku_i[..., :2*d], u_j[..., :2*d])

    # =========================================================================
    # STEP 4: Bilinear gating for V portion
    # =========================================================================
    # Per-channel K*Q gating within each block
    log_K = u_i[..., :d]       # K channels from earlier accumulation
    log_Q = u_i[..., d:2*d]    # Q channels from earlier accumulation
    log_KQ = log_K + log_Q     # Per-channel K*Q gate

    # Gate the V input and accumulate
    gated_V_input = log_KQ + u_j[..., 2*d:]
    u_new_V = log_add_exp(Ku_i[..., 2*d:], gated_V_input)

    u_new = jnp.concatenate([u_new_KQ, u_new_V], axis=-1)
    return K_new, u_new


def make_kqv_block_scan_op(block_size: int):
    """
    Create a KQV block scan operator with fixed block_size.

    JAX's associative_scan requires a binary operator, but cssm_kqv_block_scan_op
    has an extra block_size parameter. This factory creates a closure with
    block_size baked in.

    Args:
        block_size: Number of channels per block

    Returns:
        Binary scan operator suitable for jax.lax.associative_scan
    """
    def scan_op(carry_i, carry_j):
        return cssm_kqv_block_scan_op(carry_i, carry_j, block_size)
    return scan_op
