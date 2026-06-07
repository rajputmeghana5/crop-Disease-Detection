import os
import shutil
import random

SOURCE_DIR = "PlantVillage"
OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png')

random.seed(42)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

class_count = 0

for class_name in sorted(os.listdir(SOURCE_DIR)):
    class_path = os.path.join(SOURCE_DIR, class_name)

    # skip if not directory
    if not os.path.isdir(class_path):
        continue

    # skip nested dataset folder
    if class_name.lower() == "plantvillage":
        print(f" Skipped nested folder: {class_name}")
        continue

    images = []
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)

        # only valid image files
        if os.path.isfile(file_path) and file.lower().endswith(VALID_EXTENSIONS):
            images.append(file)

    if len(images) == 0:
        print(f" Skipped empty class: {class_name}")
        continue

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

    def copy_files(files, split):
        for file in files:
            src = os.path.join(class_path, file)
            dst = os.path.join(OUTPUT_DIR, split, class_name, file)
            shutil.copy2(src, dst)

    copy_files(train_imgs, "train")
    copy_files(val_imgs, "val")
    copy_files(test_imgs, "test")

    class_count += 1

    print(f" {class_name} ({total} images)")

print(f"\nTotal classes processed: {class_count}")