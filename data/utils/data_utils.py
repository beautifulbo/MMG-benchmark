# coding: utf-8

"""
Image and text data utilities
"""
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import lmdb


def image_to_tensor(image):
    """Convert numpy HWC image to PyTorch CHW tensor."""
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    tensor = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1))
    return tensor


class ImageResize:
    """Resize image to target size."""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img.resize((self.size, self.size), Image.BILINEAR)


class ImagePad:
    """Pad image to square."""
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        pad_w = (self.size - w) // 2
        pad_h = (self.size - h) // 2
        return Image.pad(img, (pad_w, pad_h, self.size - w - pad_w, self.size - h - pad_h), fill=self.fill)


def get_imagenet_transform(max_size=256):
    """Get standard imagenet transform."""
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    return Compose([
        Resize(max_size),
        CenterCrop(max_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def mask_batch_text_tokens(tokens, mask_ratio=0.8):
    """MLM masking for text tokens (80% [MASK], 10% random, 10% unchanged)."""
    masked_tokens = tokens.clone()
    mask = torch.rand_like(tokens, dtype=torch.float32) < mask_ratio
    random_mask = (torch.rand_like(tokens, dtype=torch.float32) < 0.1) & mask
    replace_with_mask = mask & ~random_mask
    masked_tokens[replace_with_mask] = 103  # [MASK] token id
    return masked_tokens, mask


def load_decompress_img_from_lmdb_value(value):
    """Load PIL image from LMDB binary blob."""
    return Image.open(BytesIO(value))


def chunk_list(lst, n):
    """Split list into n chunks."""
    return [lst[i::n] for i in range(n)]


def mk_input_group(groups, max_len):
    """Organize examples into groups for batch training."""
    result = []
    for group in groups:
        for i in range(0, len(group), max_len):
            result.append(group[i:i+max_len])
    return result