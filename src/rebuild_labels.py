from pathlib import Path
import pandas as pd

CLASS_MAP = {
    "normal": 0,
    "ocp": 1,
    "ocp_chronic": 2,
    "post_viral_ded": 3,
    "sjs": 4,
    "symblepharon": 5,
}

rows = []

dataset_root = Path("dataset")

for class_name, class_idx in CLASS_MAP.items():
    class_dir = dataset_root / class_name

    if not class_dir.exists():
        continue

    for img in class_dir.iterdir():
        if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            rows.append({
                "filepath": str(img).replace("\\", "/"),
                "class_key": class_name,
                "class_idx": class_idx,
            })

df = pd.DataFrame(rows)

df.to_csv("dataset/labels.csv", index=False)

print(f"Created labels.csv with {len(df)} rows")
print(df["class_key"].value_counts())