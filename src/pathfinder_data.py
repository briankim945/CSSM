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

Supports:
- Sequential loading (original)
- Parallel loading with ThreadPoolExecutor (--num_workers > 0)
- TFRecord format for maximum I/O performance
"""

import os
from pathlib import Path
from typing import Tuple, Iterator, Optional
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

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

        # imgs_dir = self.root / 'imgs'
        imgs_dir = self.root

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
    train_ratio: float = 0.85,
    val_ratio: float = 0.15,
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
    difficulty_dir = os.path.join(root, f'curv_contour_length_{difficulty}', 'imgs')

    if not os.path.exists(difficulty_dir):
        raise ValueError(f"Difficulty folder not found: {difficulty_dir}")
    
    train_dir = os.path.join(difficulty_dir, 'train')

    if not os.path.exists(train_dir):
        raise ValueError(f"Train folder not found: {train_dir}")
    
    test_dir = os.path.join(difficulty_dir, 'test')

    if not os.path.exists(test_dir):
        raise ValueError(f"Test folder not found: {test_dir}")

    # Create train/val dataset
    train_val_dataset = PathfinderDataset(
        root=train_dir,
        image_size=image_size,
        num_frames=num_frames,
    )

    # Create test dataset
    test_dataset = PathfinderDataset(
        root=test_dir,
        image_size=image_size,
        num_frames=num_frames,
    )

    # Split indices
    n_total = len(train_val_dataset)
    indices = np.arange(n_total)

    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_train = int(n_total * train_ratio)
    # n_val = int(n_total * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    # test_indices = indices[n_train + n_val:]

    # Create subset datasets
    train_dataset = PathfinderSubset(train_val_dataset, train_indices)
    val_dataset = PathfinderSubset(train_val_dataset, val_indices)
    # test_dataset = PathfinderSubset(full_dataset, test_indices)

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


class PathfinderVideoLoaderFast:
    """
    Fast parallel data loader for Pathfinder with prefetching.

    Uses ThreadPoolExecutor to load batches in parallel and a prefetch
    queue to overlap data loading with GPU compute.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 4,
        prefetch_batches: int = 2,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.n_samples = len(dataset)

    def __len__(self):
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def _load_single(self, idx: int) -> Tuple[np.ndarray, int]:
        """Load a single sample (called by worker threads)."""
        return self.dataset[idx]

    def _load_batch(self, batch_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Load a batch of samples using thread pool."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._load_single, batch_indices))

        images = np.stack([r[0] for r in results], axis=0)
        labels = np.array([r[1] for r in results], dtype=np.int32)
        return images, labels

    def _prefetch_worker(self, batch_indices_list: list, queue: Queue, stop_event: threading.Event):
        """Background worker that prefetches batches into queue."""
        for batch_indices in batch_indices_list:
            if stop_event.is_set():
                break
            try:
                images, labels = self._load_batch(batch_indices)
                queue.put((jnp.array(images), jnp.array(labels)))
            except Exception as e:
                print(f"Prefetch error: {e}")
                queue.put(None)
        queue.put(None)  # Signal end of data

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        # Create list of batch indices
        batch_indices_list = []
        for start_idx in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            batch_indices_list.append(batch_indices)

        # Start prefetch worker
        queue = Queue(maxsize=self.prefetch_batches)
        stop_event = threading.Event()
        worker = threading.Thread(
            target=self._prefetch_worker,
            args=(batch_indices_list, queue, stop_event)
        )
        worker.start()

        # Yield batches from queue
        try:
            while True:
                batch = queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            stop_event.set()
            worker.join(timeout=1.0)


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
    num_workers: int = 0,
    prefetch_batches: int = 2,
):
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
        num_workers: Number of parallel workers (0=sequential, >0=parallel)
        prefetch_batches: Number of batches to prefetch (only when num_workers>0)

    Returns:
        PathfinderVideoLoader or PathfinderVideoLoaderFast yielding (images, labels)
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

    if num_workers > 0:
        print(f"  Using parallel loader with {num_workers} workers, {prefetch_batches} prefetch batches")
        return PathfinderVideoLoaderFast(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
        )
    else:
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
    # n_pos = len(list(Path(difficulty_dir).glob('pos/*.png'))) if os.path.exists(os.path.join(difficulty_dir, 'pos')) else 0
    # n_neg = len(list(Path(difficulty_dir).glob('neg/*.png'))) if os.path.exists(os.path.join(difficulty_dir, 'neg')) else 0
    # total = n_pos + n_neg
    # train_size = int(total * train_ratio)

    train_n_pos = len(list(Path(difficulty_dir).glob('train/pos/*.png'))) if os.path.exists(os.path.join(difficulty_dir, 'train', 'pos')) else 0
    train_n_neg = len(list(Path(difficulty_dir).glob('train/neg/*.png'))) if os.path.exists(os.path.join(difficulty_dir, 'train', 'neg')) else 0
    test_n_pos = len(list(Path(difficulty_dir).glob('test/pos/*.png'))) if os.path.exists(os.path.join(difficulty_dir, 'test', 'pos')) else 0
    test_n_neg = len(list(Path(difficulty_dir).glob('test/neg/*.png'))) if os.path.exists(os.path.join(difficulty_dir, 'test', 'neg')) else 0
    train_size = train_n_pos + train_n_neg
    total = train_size + test_n_pos + test_n_neg

    return {
        'num_classes': 2,
        'train_size': train_size,
        'total_size': total,
        'image_size': 300,  # Original size
        'difficulty': difficulty,
        'task': 'binary_classification',
        'description': f'Pathfinder contour length {difficulty}',
    }


# =============================================================================
# TFRecord-based loader for maximum I/O performance
# =============================================================================

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


class PathfinderTFRecordLoader:
    """
    Fast TFRecord-based data loader for Pathfinder.

    Requires pre-conversion using scripts/convert_pathfinder_to_tfrecord.py
    """

    def __init__(
        self,
        tfrecord_dir: str,
        difficulty: str,
        split: str,
        batch_size: int,
        num_frames: int = 1,
        shuffle: bool = True,
        shuffle_buffer: int = 10000,
        prefetch_batches: int = 2,
    ):
        if not HAS_TF:
            raise ImportError("TensorFlow required for TFRecord loading. Install with: pip install tensorflow")

        self.batch_size = batch_size
        self.num_frames = num_frames
        self.shuffle = shuffle

        # Find TFRecord shards
        shard_dir = Path(tfrecord_dir) / f'difficulty_{difficulty}' / split
        self.shard_paths = sorted(shard_dir.glob('*.tfrecord'))

        if len(self.shard_paths) == 0:
            raise ValueError(f"No TFRecord shards found in {shard_dir}")

        # Load metadata for sample count
        metadata_path = Path(tfrecord_dir) / f'difficulty_{difficulty}' / 'metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.n_samples = metadata.get(f'{split}_samples', 0)
            self.image_size = metadata.get('image_size', 224)
        else:
            # Estimate from file count
            self.n_samples = len(self.shard_paths) * 5000  # Rough estimate
            self.image_size = 224

        # Create TF dataset
        self.dataset = self._create_dataset(shuffle_buffer, prefetch_batches)

    def _parse_example(self, serialized):
        """Parse a single TFRecord example."""
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'channels': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(serialized, features)

        # Decode image
        image = tf.io.decode_raw(example['image'], tf.float32)
        image = tf.reshape(image, [self.image_size, self.image_size, 3])

        label = tf.cast(example['label'], tf.int32)

        return image, label

    def _create_dataset(self, shuffle_buffer: int, prefetch_batches: int):
        """Create TF dataset pipeline."""
        # Interleave shards for better I/O
        files = tf.data.Dataset.from_tensor_slices([str(p) for p in self.shard_paths])

        if self.shuffle:
            files = files.shuffle(len(self.shard_paths))

        dataset = files.interleave(
            lambda x: tf.data.TFRecordDataset(x),
            cycle_length=4,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if self.shuffle:
            dataset = dataset.shuffle(shuffle_buffer)

        dataset = dataset.map(self._parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(prefetch_batches)

        return dataset

    def __len__(self):
        return self.n_samples // self.batch_size

    def __iter__(self):
        for images, labels in self.dataset:
            # Convert to numpy and add temporal dimension
            images_np = images.numpy()  # (B, H, W, 3)

            # Repeat for temporal dimension: (B, H, W, 3) -> (B, T, H, W, 3)
            images_video = np.stack([images_np] * self.num_frames, axis=1)

            yield jnp.array(images_video), jnp.array(labels.numpy())


def get_pathfinder_tfrecord_loader(
    tfrecord_dir: str,
    difficulty: str = '9',
    batch_size: int = 32,
    num_frames: int = 1,
    split: str = 'train',
    shuffle: bool = True,
    shuffle_buffer: int = 10000,
    prefetch_batches: int = 2,
):
    """
    Get TFRecord-based loader for Pathfinder.

    Requires pre-conversion: python scripts/convert_pathfinder_to_tfrecord.py

    Args:
        tfrecord_dir: Directory containing TFRecord shards
        difficulty: '9', '14', or '20'
        batch_size: Batch size
        num_frames: Number of frames for temporal dimension
        split: 'train', 'val', or 'test'
        shuffle: Whether to shuffle data
        shuffle_buffer: Size of shuffle buffer
        prefetch_batches: Number of batches to prefetch

    Returns:
        PathfinderTFRecordLoader yielding (images, labels)
    """
    loader = PathfinderTFRecordLoader(
        tfrecord_dir=tfrecord_dir,
        difficulty=difficulty,
        split=split,
        batch_size=batch_size,
        num_frames=num_frames,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        prefetch_batches=prefetch_batches,
    )

    print(f"  TFRecord loader: {len(loader)} batches from {len(loader.shard_paths)} shards")

    return loader
