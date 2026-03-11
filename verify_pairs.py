import os
import nibabel as nib
import numpy as np

# ---- UPDATE THIS PATH IF NEEDED ----
base_data_path = "data/raw"   # change if your data folder is elsewhere

# Volume folders (as shown in your screenshot)
volume_dirs = [
    os.path.join(base_data_path, "volume_pt1"),
    os.path.join(base_data_path, "volume_pt2"),
    os.path.join(base_data_path, "volume_pt3"),
    os.path.join(base_data_path, "volume_pt4"),
    os.path.join(base_data_path, "volume_pt5"),
]

# Segmentation folder
seg_dir = os.path.join(base_data_path, "segmentations")

# Get list of all segmentation files
seg_files = [
    f for f in os.listdir(seg_dir)
    if f.endswith(".nii") or f.endswith(".nii.gz")
]

total_checked = 0
total_errors = 0

for volume_dir in volume_dirs:

    if not os.path.exists(volume_dir):
        print(f"⚠ Folder not found: {volume_dir}")
        continue

    vol_files = [
        f for f in os.listdir(volume_dir)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]

    for vol_file in vol_files:
        total_checked += 1

        # Extract case number
        number = vol_file.split("-")[1].split(".")[0]

        # Try both possible segmentation extensions
        possible_seg_files = [
            f"segmentation-{number}.nii",
            f"segmentation-{number}.nii.gz"
        ]

        seg_file = None
        for candidate in possible_seg_files:
            if candidate in seg_files:
                seg_file = candidate
                break

        if seg_file is None:
            print(f"❌ Missing segmentation for {vol_file}")
            total_errors += 1
            continue

        # Load files
        vol = nib.load(os.path.join(volume_dir, vol_file))
        seg = nib.load(os.path.join(seg_dir, seg_file))

        # Shape check
        if vol.shape != seg.shape:
            print(f"❌ Shape mismatch for {vol_file}")
            print("   Volume shape:", vol.shape)
            print("   Seg shape   :", seg.shape)
            total_errors += 1
            continue

        # Affine check
        if not np.allclose(vol.affine, seg.affine):
            print(f"⚠ Affine mismatch for {vol_file}")
            total_errors += 1
            continue

        print(f"✅ {vol_file} matches {seg_file}")

print("\nVerification complete.")
print(f"Total files checked: {total_checked}")
print(f"Total errors found : {total_errors}")