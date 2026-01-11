# CSSM: Cepstral State Space Model

A JAX/Flax implementation of Cepstral State Space Models for vision tasks. CSSM replaces traditional attention mechanisms with FFT-based spectral convolutions and temporal recurrence.

## Key Features

- **FFT-based spatial processing**: Efficient O(N log N) spatial mixing via spectral convolution
- **Temporal recurrence**: Associative scan for parallel temporal processing
- **GOOM numerics**: Log-space computation for numerical stability
- **Two CSSM variants**:
  - `StandardCSSM`: Scalar recurrence with single spatial kernel
  - `GatedOpponentCSSM`: 2x2 coupled oscillator with excitation/inhibition dynamics

## Architectures

| Architecture | Description |
|-------------|-------------|
| `vit` | CSSM-ViT: ViT with CSSM replacing attention |
| `baseline` | Standard ViT baseline for comparison |
| `deit3` | DeiT3-Large baseline (1024 dim, 24 layers) |
| `cssm_deit3` | CSSM-DeiT3: DeiT3 with CSSM replacing attention |
| `convnext` | ConvNeXt-style CSSM blocks |

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd CSSM

# Create conda environment
conda create -n cssm python=3.10
conda activate cssm

# Install dependencies
pip install -r requirements.txt
```

## Datasets

### Imagenette (default)
Small 10-class subset of ImageNet for fast experimentation.
```bash
# Download Imagenette
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf imagenette2-320.tgz
```

### Pathfinder
Binary classification task testing long-range spatial integration.
```bash
python main.py --dataset pathfinder --pathfinder_difficulty 9
```

### ImageNet
Full ImageNet-1K for large-scale training.
```bash
python src/training/train_imagenet.py --model cssm_deit3
```

## Usage

### Quick Start (Imagenette)

```bash
# CSSM-ViT with opponent dynamics
python main.py --arch vit --cssm opponent --seq_len 8

# Standard CSSM (no opponent surround)
python main.py --arch vit --cssm standard

# Baseline ViT for comparison
python main.py --arch baseline
```

### DeiT3 / CSSM-DeiT3 on ImageNet

```bash
# DeiT3-Large baseline
python src/training/train_imagenet.py --model deit3 --batch_size 32

# CSSM-DeiT3-Large with opponent dynamics
python src/training/train_imagenet.py --model cssm_deit3 --cssm_type opponent --num_timesteps 8

# CSSM-DeiT3-Large with standard dynamics
python src/training/train_imagenet.py --model cssm_deit3 --cssm_type standard
```

### Multi-GPU Training

The ImageNet training script automatically uses all available GPUs via JAX pmap:
```bash
# Uses all visible GPUs
python src/training/train_imagenet.py --model cssm_deit3 --batch_size 32

# Specify GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/training/train_imagenet.py --model cssm_deit3
```

### Pathfinder Experiments

```bash
# Easy difficulty (contour length 9)
python main.py --arch vit --dataset pathfinder --pathfinder_difficulty 9

# Hard difficulty (contour length 20)
python main.py --arch vit --dataset pathfinder --pathfinder_difficulty 20

# Deep temporal recurrence
python main.py --arch vit --depth 1 --seq_len 96 --dataset pathfinder
```

## Key Arguments

### Architecture
| Argument | Options | Description |
|----------|---------|-------------|
| `--arch` | `vit`, `baseline`, `deit3`, `cssm_deit3`, `convnext` | Model architecture |
| `--cssm` | `standard`, `opponent` | CSSM variant |
| `--embed_dim` | int | Embedding dimension (384=small, 768=base, 1024=large) |
| `--depth` | int | Number of transformer blocks |
| `--seq_len` | int | Number of timesteps for CSSM recurrence |

### Training
| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 8 | Per-device batch size |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 1e-4 | Peak learning rate |
| `--weight_decay` | 1e-4 | Weight decay |
| `--grad_clip` | 1.0 | Gradient clipping norm |

### Data
| Argument | Options | Description |
|----------|---------|-------------|
| `--dataset` | `imagenette`, `pathfinder`, `imagenet` | Dataset |
| `--image_size` | 224, 384 | Input image size |
| `--data_dir` | path | Dataset directory |

## Project Structure

```
CSSM/
├── main.py                     # Main training script (Imagenette/Pathfinder)
├── requirements.txt
├── README.md
├── src/
│   ├── data.py                 # Imagenette data loading
│   ├── pathfinder_data.py      # Pathfinder data loading
│   ├── data/
│   │   └── imagenet.py         # ImageNet streaming loader
│   ├── models/
│   │   ├── cssm.py             # Core CSSM implementations
│   │   ├── cssm_vit.py         # CSSM-ViT architecture
│   │   ├── baseline_vit.py     # Standard ViT baseline
│   │   ├── deit3.py            # DeiT3-Large baseline
│   │   ├── cssm_deit3.py       # CSSM-DeiT3-Large
│   │   ├── goom.py             # GOOM log-space primitives
│   │   ├── math.py             # Scan operations
│   │   └── convnext.py         # ConvNeXt-style models
│   └── training/
│       ├── distributed.py      # Multi-GPU utilities
│       └── train_imagenet.py   # ImageNet training script
├── benchmark_timing.py         # Timing comparison script
├── run_ablations.sh            # Ablation study script
└── visualize_filters.py        # Filter visualization
```

## CSSM Architecture Details

### GatedOpponentCSSM (Recommended)

Implements a 2x2 coupled oscillator with:
- **Diagonal**: Decay terms (alpha for X, delta for Y)
- **Off-diagonal**: Coupling (mu for inhibition X->Y, gamma for excitation Y->X)
- **Gating**: Input-dependent gates control dynamics
- **Output**: Concatenate [X, Y] and project back to C channels

```
State transition:
[X_t]   [alpha  -K_I*mu ] [X_{t-1}]   [U_t]
[Y_t] = [K_E*gamma delta] [Y_{t-1}] + [0  ]
```

### StandardCSSM

Simple scalar recurrence:
```
H_t = K * H_{t-1} + U_t  (in spectral domain)
```

### GOOM (Generalized Order of Magnitude)

Log-space representation for numerical stability:
- Prevents gradient underflow/overflow in deep temporal scans
- Custom VJPs for log/exp operations
- Handles negative values via phase encoding

## Benchmarking

Compare timing across architectures:
```bash
python benchmark_timing.py --depth 12 --seq_len 8 --num_steps 100
```

Run ablation studies:
```bash
# Full ablation (50 configurations)
./run_ablations.sh

# Custom configuration
GPU=0 EPOCHS=50 ./run_ablations.sh
```

## Citation

If you use this code, please cite:
```bibtex
@article{cssm2024,
  title={Cepstral State Space Models for Vision},
  author={...},
  year={2024}
}
```

## License

MIT License
