"""CSSM and ConvNeXt model components."""

from .cssm import StandardCSSM, GatedOpponentCSSM
from .convnext import ConvNextBlock, CSSMNextBlock, HybridBlock, ModelFactory
from .cssm_vit import CSSMViT, CSSMBlock, cssm_vit_tiny, cssm_vit_small, cssm_vit_base
from .goom import to_goom, from_goom, goom_log, goom_exp, goom_abs, GOOMConfig
from .operations import log_add_exp, log_sum_exp, log_matmul_2x2, log_matvec_2x2
from .math import cssm_scalar_scan_op, cssm_matrix_scan_op

__all__ = [
    # CSSM layers
    "StandardCSSM",
    "GatedOpponentCSSM",
    # ConvNeXt blocks
    "ConvNextBlock",
    "CSSMNextBlock",
    "HybridBlock",
    "ModelFactory",
    # CSSM-ViT models
    "CSSMViT",
    "CSSMBlock",
    "cssm_vit_tiny",
    "cssm_vit_small",
    "cssm_vit_base",
    # GOOM primitives
    "to_goom",
    "from_goom",
    "goom_log",
    "goom_exp",
    "goom_abs",
    "GOOMConfig",
    # Log-space operations
    "log_add_exp",
    "log_sum_exp",
    "log_matmul_2x2",
    "log_matvec_2x2",
    # Scan operators
    "cssm_scalar_scan_op",
    "cssm_matrix_scan_op",
]
