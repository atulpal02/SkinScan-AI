import os
import shutil
import random

SOURCE = "data/raw"
TRAIN = "data/processed/train"
VAL = "data/processed/val"

SPLIT_RATIO = 0.8

for cls in os.listdir(SOURCE):
    cls_path = os.path.join(SOURCE, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATIO)

    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    os.makedirs(os.path.join(TRAIN, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(TRAIN, cls, img))

    for img in val_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(VAL, cls, img))

print(" Data split done")