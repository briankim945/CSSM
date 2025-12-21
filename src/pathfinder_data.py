"""
Pathfinder dataset loader for CSSM models.

Pathfinder is a binary classification task: determine if two marked dots
are connected by a contour path. This tests long-range spatial integration
where standard CNNs typically fail (chance = 50%).

The dataset has three difficulty levels based on contour length:
- 9: Easiest (shortest contours)
- 14: Medium
- 20: Hardest (longest contours)

For CSSM models that expect video input (B, T, H, W, C), we treat static
images as single-frame "videos" with T=1, or optionally repeat frames.
"""

import os
from pathlib import Path
from typing import Tuple, Iterator, Optional

import numpy as np
from PIL import Image
import jax.numpy as jnp

# ImageNet normalization constants (same as Imagenette)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PathfinderDataset:
    """
    Pathfinder dataset with pos/neg binary classification.

    Attributes:
        root: Path to difficulty folder (e.g., curv_contour_length_9)
        image_paths: List of image file paths
        labels: List of labels (0=neg, 1=pos)
        image_size: Target image size (default 224 for ViT compatibility)
        num_frames: Number of times to repeat the image for temporal dim
    """

    def __init__(
        self,
        root: str,
        image_size: int = 224,
        num_frames: int = 1,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.num_frames = num_frames

        # Collect images from pos and neg folders
        self.image_paths = []
        self.labels = []

        imgs_dir = self.root / 'imgs'

        # Negative examples (label=0)
        neg_dir = imgs_dir / 'neg'
        if neg_dir.exists():
            for img_path in sorted(neg_dir.glob('*.png')):
                self.image_paths.append(img_path)
                self.labels.append(0)

        # Positive examples (label=1)
        pos_dir = imgs_dir / 'pos'
        if pos_dir.exists():
            for img_path in sorted(pos_dir.glob('*.png')):
                self.image_paths.append(img_path)
                self.labels.append(1)

        self.labels = np.array(self.labels)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {imgs_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Load and preprocess a single image."""
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load grayscale image
        img = Image.open(img_path).convert('L')

        # Resize to target size
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to numpy and normalize to [0, 1]
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Convert grayscale to RGB by repeating channels
        img_rgb = np.stack([img_np, img_np, img_np], axis=-1)  # (H, W, 3)

        # Apply ImageNet normalization (same as Imagenette)
        img_rgb = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD

        # Add temporal dimension: (H, W, 3) -> (T, H, W, 3)
        # Always repeat to num_frames (same as Imagenette loader)
        img_video = np.stack([img_rgb] * self.num_frames, axis=0)

        return img_video, label


def get_pathfinder_datasets(
    root: str = '/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025',
    difficulty: str = '9',
    image_size: int = 224,
    num_frames: int = 1,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple['PathfinderDataset', 'PathfinderDataset', 'PathfinderDataset']:
    """
    Get train/val/test splits for Pathfinder dataset.

    Args:
        root: Root directory containing difficulty folders
        difficulty: '9', '14', or '20' (contour length)
        image_size: Target image size
        num_frames: Number of frames for temporal dimension
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    difficulty_dir = os.path.join(root, f'curv_contour_length_{difficulty}')

    if not os.path.exists(difficulty_dir):
        raise ValueError(f"Difficulty folder not found: {difficulty_dir}")

    # Create full dataset
    full_dataset = PathfinderDataset(
        root=difficulty_dir,
        image_size=image_size,
        num_frames=num_frames,
    )

    # Split indices
    n_total = len(full_dataset)
    indices = np.arange(n_total)

    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Create subset datasets
    train_dataset = PathfinderSubset(full_dataset, train_indices)
    val_dataset = PathfinderSubset(full_dataset, val_indices)
    test_dataset = PathfinderSubset(full_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


class PathfinderSubset:
    """Subset of PathfinderDataset using specific indices."""

    def __init__(self, dataset: PathfinderDataset, indices: np.ndarray):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.dataset[self.indices[idx]]


class PathfinderVideoLoader:
    """
    Data loader for Pathfinder that matches Imagenette's VideoDataLoader interface.

    Provides __len__ for proper tqdm progress display.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_samples = len(dataset)

    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]

            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            images = []
            labels = []

            for idx in batch_indices:
                img, label = self.dataset[idx]
                images.append(img)
                labels.append(label)

            images = np.stack(images, axis=0)  # (B, T, H, W, 3)
            labels = np.array(labels, dtype=np.int32)

            yield jnp.array(images), jnp.array(labels)


def get_pathfinder_loader(
    root: str = '/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025',
    difficulty: str = '9',
    batch_size: int = 32,
    image_size: int = 224,
    num_frames: int = 1,
    split: str = 'train',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
) -> PathfinderVideoLoader:
    """
    Get a batch iterator for Pathfinder dataset.

    Args:
        root: Root directory
        difficulty: '9', '14', or '20'
        batch_size: Batch size
        image_size: Target image size
        num_frames: Number of frames for temporal dimension
        split: 'train', 'val', or 'test'
        train_ratio: Train split ratio
        val_ratio: Val split ratio
        seed: Random seed
        shuffle: Whether to shuffle data

    Returns:
        PathfinderVideoLoader yielding (images, labels) tuples
        images: (B, T, H, W, 3) float32
        labels: (B,) int32
    """
    train_ds, val_ds, test_ds = get_pathfinder_datasets(
        root=root,
        difficulty=difficulty,
        image_size=image_size,
        num_frames=num_frames,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    if split == 'train':
        dataset = train_ds
    elif split == 'val':
        dataset = val_ds
    else:
        dataset = test_ds

    print(f"  Loaded {len(dataset)} {split} samples")

    return PathfinderVideoLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )


def get_pathfinder_info(
    difficulty: str = '9',
    root: str = '/media/data_cifs_lrs/projects/prj_LRA/PathFinder/pathfinder300_new_2025',
    train_ratio: float = 0.8,
) -> dict:
    """Get dataset metadata."""
    # Count actual images
    difficulty_dir = os.path.join(root, f'curv_contour_length_{difficulty}', 'imgs')
    n_pos = len(list(Path(difficulty_dir).glob('pos/*.png'))) if os.path.exists(os.path.join(difficulty_dir, 'pos')) else 0
    n_neg = len(list(Path(difficulty_dir).glob('neg/*.png'))) if os.path.exists(os.path.join(difficulty_dir, 'neg')) else 0
    total = n_pos + n_neg
    train_size = int(total * train_ratio)

    return {
        'num_classes': 2,
        'train_size': train_size,
        'total_size': total,
        'image_size': 300,  # Original size
        'difficulty': difficulty,
        'task': 'binary_classification',
        'description': f'Pathfinder contour length {difficulty}',
    }
