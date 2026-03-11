import nibabel as nib
import numpy as np
vol = nib.load("/Users/arjunkhimta/Desktop/Codes/E-LilNet/data/raw/volume_pt3/volume-25.nii")
seg = nib.load("/Users/arjunkhimta/Desktop/Codes/E-LilNet/data/raw/segmentations/segmentation-25.nii")

print(vol.shape)
print(seg.shape)
seg_data = seg.get_fdata()

print("Unique labels in segmentation:", np.unique(seg_data))
if 2 in np.unique(seg_data):
    print("Tumor present")
else:
    print("No tumor present")