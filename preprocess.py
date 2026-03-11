import os
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

# ==============================
# PATH CONFIGURATION
# ==============================

BASE_RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

VOLUME_DIRS = [
    os.path.join(BASE_RAW_PATH, "volume_pt1"),
    os.path.join(BASE_RAW_PATH, "volume_pt2"),
    os.path.join(BASE_RAW_PATH, "volume_pt3"),
    os.path.join(BASE_RAW_PATH, "volume_pt4"),
    os.path.join(BASE_RAW_PATH, "volume_pt5"),
]

SEG_DIR = os.path.join(BASE_RAW_PATH, "segmentations")

IMG_SAVE_DIR = os.path.join(PROCESSED_PATH, "images")
MASK_SAVE_DIR = os.path.join(PROCESSED_PATH, "masks")

os.makedirs(IMG_SAVE_DIR, exist_ok=True)
os.makedirs(MASK_SAVE_DIR, exist_ok=True)

# ==============================
# PARAMETERS
# ==============================

TARGET_SIZE = 256
HU_MIN = -200
HU_MAX = 250

slice_counter = 0

# ==============================
# PROCESSING LOOP
# ==============================

for volume_dir in VOLUME_DIRS:

    vol_files = [
        f for f in os.listdir(volume_dir)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]

    for vol_file in tqdm(vol_files, desc=f"Processing {volume_dir}"):

        case_id = vol_file.split("-")[1].split(".")[0]

        seg_file = f"segmentation-{case_id}.nii"
        if not os.path.exists(os.path.join(SEG_DIR, seg_file)):
            seg_file = f"segmentation-{case_id}.nii.gz"

        seg_path = os.path.join(SEG_DIR, seg_file)
        vol_path = os.path.join(volume_dir, vol_file)

        if not os.path.exists(seg_path):
            print(f"Skipping {case_id} (no segmentation found)")
            continue

        # Load volume and mask
        volume = nib.load(vol_path).get_fdata()
        mask = nib.load(seg_path).get_fdata()

        depth = volume.shape[2]

        for i in range(depth):

            img_slice = volume[:, :, i]
            mask_slice = mask[:, :, i]

            # Skip completely empty slices
            if np.sum(mask_slice) == 0:
                continue

            # Clip HU values
            img_slice = np.clip(img_slice, HU_MIN, HU_MAX)

            # Normalize to 0-1
            img_slice = (img_slice - HU_MIN) / (HU_MAX - HU_MIN)

            # Resize image
            img_slice = cv2.resize(
                img_slice,
                (TARGET_SIZE, TARGET_SIZE),
                interpolation=cv2.INTER_LINEAR
            )

            # Resize mask (nearest neighbor to preserve labels)
            mask_slice = cv2.resize(
                mask_slice,
                (TARGET_SIZE, TARGET_SIZE),
                interpolation=cv2.INTER_NEAREST
            )

            # Convert mask to int
            mask_slice = mask_slice.astype(np.uint8)

            # Save
            np.save(
                os.path.join(IMG_SAVE_DIR, f"image_{slice_counter:05d}.npy"),
                img_slice.astype(np.float32)
            )

            np.save(
                os.path.join(MASK_SAVE_DIR, f"mask_{slice_counter:05d}.npy"),
                mask_slice
            )

            slice_counter += 1

print("\nPreprocessing Complete.")
print(f"Total saved slices: {slice_counter}")