import os
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

RAW_PATH = "data/raw"
SEG_PATH = os.path.join(RAW_PATH, "segmentations")
INTERIM_PATH = "data/interim"
SPLIT_PATH = "experiments"

def normalize_ct(image):
    image = np.clip(image, -100, 400)
    image = (image + 100) / 500
    return image

def load_ids(split_name):
    with open(os.path.join(SPLIT_PATH, f"{split_name}_ids.txt"), "r") as f:
        ids = f.read().splitlines()
    return ids

def find_volume_path(patient_id):
    filename = f"volume-{patient_id}.nii"
    for folder in os.listdir(RAW_PATH):
        if folder.startswith("volume_pt"):
            candidate = os.path.join(RAW_PATH, folder, filename)
            if os.path.exists(candidate):
                return candidate
    return None

def process_split(split_name):
    ids = load_ids(split_name)
    output_dir = os.path.join(INTERIM_PATH, split_name)
    os.makedirs(output_dir, exist_ok=True)

    for pid in tqdm(ids):
        volume_path = find_volume_path(pid)
        mask_path = os.path.join(SEG_PATH, f"segmentation-{pid}.nii")

        if volume_path is None or not os.path.exists(mask_path):
            continue

        volume = nib.load(volume_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        depth = volume.shape[2]

        for i in range(depth):
            img_slice = volume[:, :, i]
            mask_slice = mask[:, :, i]

            # Keep only tumor slices
            if np.any(mask_slice == 2):

                img_slice = normalize_ct(img_slice)

                img_slice = cv2.resize(img_slice, (256, 256))
                mask_slice = cv2.resize(mask_slice, (256, 256), interpolation=cv2.INTER_NEAREST)

                np.save(os.path.join(output_dir, f"{pid}_img_{i}.npy"), img_slice)
                np.save(os.path.join(output_dir, f"{pid}_mask_{i}.npy"), mask_slice)

def main():
    for split in ["train", "val", "test"]:
        print(f"Processing {split} set...")
        process_split(split)

if __name__ == "__main__":
    main()