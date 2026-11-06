
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "mobilenetv2_split.csv"

if not MANIFEST.exists():
    raise FileNotFoundError(f"Split manifest not found: {MANIFEST}")

df = pd.read_csv(MANIFEST)

print("\nCSV columns:", list(df.columns))
print("\nFirst five rows:")
print(df.head().to_string(index=False))

# Identify likely label, split, and image-path columns.
def find_column(candidates):
    for col in df.columns:
        if col.lower() in candidates:
            return col
    return None

label_col = find_column({"label", "class", "category", "target"})
split_col = find_column({"split", "partition", "set", "subset"})
path_col = find_column({"path", "filepath", "file_path", "image_path", "filename"})

if not all([label_col, split_col, path_col]):
    print(
        "\nCould not automatically identify all required columns."
        "\nCheck the column names above and update the script."
    )
    raise SystemExit(1)

print("\n=== Images per class and split ===")
print(pd.crosstab(df[label_col], df[split_col], margins=True))

# Extract the 14-digit timestamp used in dataset filenames.
def extract_timestamp(value):
    match = re.search(r"(\d{14})", str(value))
    return pd.to_datetime(
        match.group(1), format="%Y%m%d%H%M%S", errors="coerce"
    ) if match else pd.NaT

df["_timestamp"] = df[path_col].apply(extract_timestamp)

print("\n=== Timestamp ranges by class and split ===")
for (label, split), group in df.groupby([label_col, split_col]):
    times = group["_timestamp"].dropna()
    if times.empty:
        print(f"{label} | {split}: no timestamps parsed")
    else:
        print(
            f"{label} | {split}: "
            f"{len(group)} images | "
            f"{times.min()} to {times.max()}"
        )

print("\n=== Audit interpretation ===")
print(
    "If timestamps from the same class overlap across train, validation, "
    "and test, nearby frames may occur in multiple splits."
)
print(
    "Timestamp overlap alone does not prove leakage, but it means the "
    "test set should not be treated as independent."
)
