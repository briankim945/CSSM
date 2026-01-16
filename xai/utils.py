import numpy as np
from scipy.ndimage import zoom
from PIL import Image
from random import randint, choice

from src.data import IMAGENET_MEAN, IMAGENET_STD


def create_image_pan_seq(
    img_path: str,
    size: int = 224,
    seq_len: int = 8,
    channels: int = 3,
    jump: int = 10,
):
    img = Image.open(img_path).convert('RGB')

    lr = randint(0, 1)
    ud = randint(0, 1)

    w, h = img.size

    master_size = seq_len * jump + size

    scale = master_size / min(w, h) * 1.15  # Slight over-scale for crop
    new_w, new_h = int(w * scale), int(h * scale)

    # Center crop
    left = (new_w - size) // 2
    top = (new_h - size) // 2

    pan_sequence = np.zeros((seq_len, size, size, channels), dtype=np.float32)

    left, h_jump = choice(
        [
            (left - (seq_len // 2) * jump, jump),
            (left + (seq_len // 2) * jump, -jump)
        ]
    )
    top, v_jump = choice(
        [
            (top - (seq_len // 2) * jump, jump),
            (top + (seq_len // 2) * jump, -jump)
        ]
    )

    img_crops = []

    # fig, axes = plt.subplots(2, 4, figsize=(5*4, 5*2))

    for i in range(seq_len):
        cur_left = left + h_jump * i
        cur_top = top + v_jump * i
        # print(cur_left, cur_top, cur_left + size, cur_top + size)
        img_crop = img.crop((cur_left, cur_top, cur_left + size, cur_top + size))
        img_crops.append(img_crop)
        pan_sequence[i] = np.array(img_crop, dtype=np.float32)
        # axes[i // 4, i % 4].imshow(img_crop)

    # plt.show()

    # Convert to array and normalize
    pan_sequence = pan_sequence / 255.0
    pan_sequence = (pan_sequence - IMAGENET_MEAN) / IMAGENET_STD

    return pan_sequence, np.stack(img_crops, axis=0)


def prep_image(image, flatten_channels=False):
    if flatten_channels:
        image = np.mean(np.abs(image), axis=-1)
    # Normalize the gradient values to be between 0-1
    max_val= np.max(image)
    min_val = np.min(image)
    image = (image - min_val) / (max_val - min_val)
    # Convert the grads to uint8 for displaying
    image = np.uint8(image * 255)
    return image  
