# Multi-GPU Training Guide

CSSM supports multi-GPU training via JAX's `pmap` for data parallelism.

## Automatic Detection

Multi-GPU training is enabled automatically when multiple GPUs are visible:

```bash
# Uses all visible GPUs
python main.py --arch vit --cssm hgru_bi ...

# Specify which GPUs to use
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py ...
```

## How It Works

1. **Data parallelism**: Batch is split across GPUs
2. **Gradient synchronization**: `pmean` averages gradients across devices
3. **State replication**: Model parameters are replicated to all devices

## Batch Size

The effective batch size is `batch_size * num_gpus`:

```bash
# With 4 GPUs and batch_size=64, effective batch = 256
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --batch_size 64 ...
```

## Complex Gradient Handling

CSSM uses FFT which produces complex-valued tensors. Since NCCL doesn't support complex types directly, gradients are converted to real representation for synchronization:

```python
# Handled automatically by pmean_complex_safe()
# Complex (a + bi) -> Real [a, b] -> pmean -> Complex (a + bi)
```

## Troubleshooting

### "NCCL error"

Complex gradients require special handling. This should be automatic, but if you see NCCL errors:

1. Ensure you're using the latest code with `pmean_complex_safe()`
2. Try reducing batch size
3. Check CUDA/NCCL versions match

### Uneven GPU utilization

All GPUs should show similar utilization. If not:

```bash
# Check GPU status
nvidia-smi -l 1

# Ensure batch_size is divisible by num_gpus
# batch_size=64 with 4 GPUs = 16 per GPU
```

### Out of Memory

With multi-GPU, each GPU holds a copy of the model:

```bash
# Reduce per-GPU batch size
--batch_size 32  # instead of 64

# Enable bf16 to reduce memory
--bf16
```

### Slow multi-GPU

1. **NVLink**: GPUs connected via NVLink are faster than PCIe
2. **Checkpointing**: Use local storage, not NFS:
   ```bash
   --checkpoint_dir /local/scratch/checkpoints
   ```

## Verification

Check that multi-GPU is working:

```python
import jax
print(f"Devices: {jax.devices()}")
print(f"Num devices: {len(jax.devices())}")
```

During training, you should see:
```
Running on 4 devices (multi-GPU enabled)
```

## Single vs Multi-GPU Performance

| GPUs | Batch | Effective Batch | Relative Speed |
|------|-------|-----------------|----------------|
| 1 | 64 | 64 | 1x |
| 2 | 64 | 128 | ~1.9x |
| 4 | 64 | 256 | ~3.7x |
| 8 | 64 | 512 | ~7.2x |

Scaling efficiency is typically 90-95% with NVLink.
