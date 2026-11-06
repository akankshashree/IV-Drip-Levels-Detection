
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "near_duplicate_pairs.csv"

df = pd.read_csv(CSV)
different = df[df["label_a"] != df["label_b"]].copy()
different = different.sort_values("hash_distance")

print("Different-label near-duplicate pairs:", len(different))
print("\nClosest 15 pairs:\n")

for row in different.head(15).itertuples(index=False):
    print(f"Hash distance: {row.hash_distance}")
    print(f"Split A: {row.split_a} | Label A: {row.label_a}")
    print(f"Split B: {row.split_b} | Label B: {row.label_b}")
    print(f"Image A: {row.path_a}")
    print(f"Image B: {row.path_b}")
    print("-" * 60)

output = ROOT / "results" / "different_label_pairs.csv"
different.to_csv(output, index=False)
print(f"\nFull report saved to: {output}")
