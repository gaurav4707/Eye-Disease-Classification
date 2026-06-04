from pathlib import Path

dataset_dir = Path("dataset")

for class_dir in dataset_dir.iterdir():
    if not class_dir.is_dir():
        continue

    images = sorted([
        f for f in class_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ])

    for i, img in enumerate(images, start=1):
        new_name = f"{class_dir.name}_{i:02d}{img.suffix.lower()}"
        img.rename(class_dir / new_name)

print("Renaming complete.")