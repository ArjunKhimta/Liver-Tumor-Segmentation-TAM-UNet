import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import sys

# Usage: python view_nii.py path_to_file.nii

file_path = sys.argv[1]

# Load NIfTI file
nii = nib.load(file_path)
data = nii.get_fdata()

print("Shape:", data.shape)

# Show middle slice
slice_index = data.shape[2] // 2
slice_img = data[:, :, slice_index]

plt.figure()
plt.imshow(np.rot90(slice_img), cmap="gray")
plt.title(f"Slice {slice_index}")
plt.show()
print("Number of slices:", data.shape[2])