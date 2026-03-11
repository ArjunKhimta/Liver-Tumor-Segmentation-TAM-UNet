import os
import random

SEG_PATH = "data/raw/segmentations"
OUTPUT_PATH = "experiments"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def main():
    files = sorted(os.listdir(SEG_PATH))
    
    # Extract patient IDs
    patient_ids = [f.replace(".nii", "").replace("segmentation-", "") for f in files]
    
    random.seed(42)  # reproducibility
    random.shuffle(patient_ids)
    
    total = len(patient_ids)
    train_end = int(0.7 * total)
    val_end = int(0.9 * total)
    
    train_ids = patient_ids[:train_end]
    val_ids = patient_ids[train_end:val_end]
    test_ids = patient_ids[val_end:]
    
    with open(f"{OUTPUT_PATH}/train_ids.txt", "w") as f:
        f.write("\n".join(train_ids))
        
    with open(f"{OUTPUT_PATH}/val_ids.txt", "w") as f:
        f.write("\n".join(val_ids))
        
    with open(f"{OUTPUT_PATH}/test_ids.txt", "w") as f:
        f.write("\n".join(test_ids))
    
    print("Total patients:", total)
    print("Train:", len(train_ids))
    print("Validation:", len(val_ids))
    print("Test:", len(test_ids))

if __name__ == "__main__":
    main()