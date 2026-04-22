import os
import shutil
import random

# Source folder
source_dir = "datasets"

# Output folders
train_dir = "dataset/train"
val_dir = "dataset/val"

# Split ratio
split_ratio = 0.8

# Create base folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Loop through categories: ai / real
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)

    if not os.path.isdir(category_path):
        continue

    all_images = []

    # Go inside animals / nature / city
    for subfolder in os.listdir(category_path):
        subfolder_path = os.path.join(category_path, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)

            # Only include files (skip folders)
            if os.path.isfile(file_path):
                all_images.append(file_path)

    # Shuffle and split
    random.shuffle(all_images)
    split_index = int(len(all_images) * split_ratio)

    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    # Create output class folders: train/ai, train/real
    train_class_dir = os.path.join(train_dir, category)
    val_class_dir = os.path.join(val_dir, category)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # Copy training images
    for src in train_images:
        dst = os.path.join(train_class_dir, os.path.basename(src))
        shutil.copy(src, dst)

    # Copy validation images
    for src in val_images:
        dst = os.path.join(val_class_dir, os.path.basename(src))
        shutil.copy(src, dst)

    print(f"{category}: {len(train_images)} train, {len(val_images)} val")

print("Dataset successfully split!")