# cABC (Contour ABC) Training Guide

The cABC task is similar to Pathfinder but uses letter-shaped contours (A, B, C) embedded in clutter.

## Dataset

cABC consists of images with:
- A letter-shaped contour (A, B, or C)
- Background clutter and distractors
- Task: classify which letter is present (3-way classification)

**Difficulties**:
- `easy`: Clear contours, minimal clutter
- `medium`: Moderate clutter (default)
- `hard`: Dense clutter, challenging

## Data Preparation

### Option 1: TFRecords (Recommended)

```bash
# Convert to TFRecords
python scripts/convert_cabc_tfrecords.py \
    --input_dir /path/to/cabc/medium \
    --output_dir /path/to/cabc_tfrecords/medium
```

### Option 2: PNG Files

Direct loading from the original dataset directory.

## Training Commands

### Recommended Configuration

```bash
python main.py --arch simple --cssm hgru_bi --dataset cabc \
    --cabc_difficulty medium \
    --tfrecord_dir /path/to/cabc_tfrecords/medium \
    --batch_size 256 --seq_len 8 --depth 1 --embed_dim 32 \
    --kernel_size 11 --lr 3e-4 --epochs 60 \
    --pos_embed spatiotemporal --bf16
```

### Without TFRecords

```bash
python main.py --arch simple --cssm hgru_bi --dataset cabc \
    --cabc_difficulty medium \
    --data_dir /path/to/cabc \
    --batch_size 256 --seq_len 8 --depth 1 --embed_dim 32 \
    --kernel_size 11 --lr 3e-4 --epochs 60
```

## Key Hyperparameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `--arch` | `simple` | Minimal architecture |
| `--cssm` | `hgru_bi` | 3x3 opponent dynamics |
| `--depth` | `1` | Single CSSM block |
| `--embed_dim` | `32` | Small dimension |
| `--seq_len` | `8` | Temporal steps |
| `--kernel_size` | `11` | Spatial kernel |
| `--batch_size` | `256` | Adjust for GPU memory |

## Expected Results

| Difficulty | Expected Accuracy |
|------------|-------------------|
| easy | ~95%+ |
| medium | ~85%+ |
| hard | ~75%+ |

## Comparison with Pathfinder

| Aspect | Pathfinder | cABC |
|--------|------------|------|
| Classes | 2 (connected/not) | 3 (A/B/C) |
| Contour shape | Random curves | Letter shapes |
| Spatial reasoning | Path following | Shape recognition |
