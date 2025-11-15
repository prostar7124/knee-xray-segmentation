import os
import numpy as np
from torch.utils.data import Dataset
import cv2

TARGET_SIZE = (224, 224)


def read_xray(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if xray is None:
        raise FileNotFoundError(f"❌ Could not read X-ray image: {path}")

    xray = cv2.resize(xray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    xray = xray.astype(np.float32) / 255.0
    xray = xray.reshape((1, *xray.shape))
    return xray


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(np.float32) / 255.0
    mask = mask.reshape((1, *mask.shape))
    return mask


class Knee_dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = str(row["xrays"]).strip()
        mask_path = str(row["masks"]).strip()

        image = read_xray(image_path)
        mask = read_mask(mask_path)

        has_mask = mask is not None

        if not has_mask:
            mask = np.zeros_like(image, dtype=np.float32)

        return {
            "image": image,
            "mask": mask,
            "has_mask": has_mask
        }
