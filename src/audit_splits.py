
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

FILES = [
    "baseline_split_manifest.csv",
    "mobilenetv2_split.csv",
    "temporal_split.csv",
]

for filename in FILES:
    path = RESULTS / filename
    print(f"\n{'=' * 65}\n{filename}")

    if not path.exists():
        print("Missing manifest")
        continue

    df = pd.read_csv(path)
    print("Columns:", list(df.columns))
    print("Rows:", len(df))

    if not {"path", "label", "split"}.issubset(df.columns):
        print("Unexpected manifest schema; skipping.")
        continue

    print("\nImages per class and split:")
    print(pd.crosstab(df["label"], df["split"]).to_string())

    print("\nImages per split:")
    print(df["split"].value_counts().to_string())

    print("\nRepeated paths across different splits:")
    path_splits = df.groupby("path")["split"].nunique()
    print(int((path_splits > 1).sum()))

    print("\nUnique paths:", df["path"].nunique())
    print("Missing files:", sum(
        not Path(p).is_file() for p in df["path"]
    ))
