# ImageNet Training Guide

Training CSSM-ViT on ImageNet-1K for large-scale image classification.

## Dataset

ImageNet-1K:
- 1.28M training images
- 50K validation images
- 1000 classes

## Training Commands

### CSSM-ViT Small (384 dim, 12 layers)

```bash
python main.py --arch vit --cssm hgru_bi --dataset imagenet \
    --imagenet_dir /path/to/imagenet \
    --batch_size 256 --seq_len 8 --depth 12 --embed_dim 384 \
    --kernel_size 11 --lr 1e-4 --epochs 300 \
    --pos_embed spatiotemporal --bf16 \
    --checkpoint_dir /local/scratch/checkpoints
```

### CSSM-ViT Base (768 dim, 12 layers)

```bash
python main.py --arch vit --cssm hgru_bi --dataset imagenet \
    --imagenet_dir /path/to/imagenet \
    --batch_size 128 --seq_len 8 --depth 12 --embed_dim 768 \
    --kernel_size 11 --lr 5e-5 --epochs 300 \
    --bf16 --checkpoint_dir /local/scratch/checkpoints
```

## Architecture Options

| Size | embed_dim | depth | num_heads (baseline) |
|------|-----------|-------|----------------------|
| Tiny | 192 | 12 | 3 |
| Small | 384 | 12 | 6 |
| Base | 768 | 12 | 12 |

## Key Hyperparameters

| Parameter | Small | Base | Notes |
|-----------|-------|------|-------|
| `--embed_dim` | 384 | 768 | Model width |
| `--depth` | 12 | 12 | Number of blocks |
| `--batch_size` | 256 | 128 | Per-GPU batch size |
| `--lr` | 1e-4 | 5e-5 | Lower for larger models |
| `--epochs` | 300 | 300 | Full training |
| `--seq_len` | 8 | 8 | Temporal steps |
| `--bf16` | Yes | Yes | Essential for speed |

## Stem Configuration

| Option | Description |
|--------|-------------|
| `--stem_mode conv` | Single conv + GELU + norm (default) |
| `--stem_mode patch` | ViT-style patch embedding |
| `--stem_stride 4` | Downsampling factor |
| `--stem_norm layer` | LayerNorm (default) or batch |

## Multi-GPU Training

See [multigpu.md](multigpu.md) for multi-GPU setup.

```bash
# All visible GPUs used automatically
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --arch vit ...
```

## Checkpointing

Use local storage for checkpoints to avoid NFS issues:

```bash
--checkpoint_dir /local/scratch/checkpoints
```

Resume from checkpoint:

```bash
--resume /local/scratch/checkpoints/run_name/epoch_100
```

## Expected Results

| Model | Top-1 Accuracy | Parameters |
|-------|----------------|------------|
| CSSM-ViT-S | ~78-80% | ~22M |
| CSSM-ViT-B | ~81-83% | ~86M |

## Comparison Baseline

Run standard ViT for comparison:

```bash
python main.py --arch baseline --dataset imagenet \
    --embed_dim 384 --depth 12 --num_heads 6 \
    --batch_size 256 --lr 1e-4 --epochs 300 --bf16
```
