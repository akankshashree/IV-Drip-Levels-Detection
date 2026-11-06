
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SOURCE = RESULTS / "mobilenetv2_split.csv"
OUTPUT = RESULTS / "grouped_split_manifest.csv"
PAIR_REPORT = RESULTS / "near_duplicate_pairs.csv"

SEED = 42
HAMMING_LIMIT = 5

df = pd.read_csv(SOURCE).reset_index(drop=True)
pairs = pd.read_csv(PAIR_REPORT)

required = {"path", "label"}
if not required.issubset(df.columns):
    raise ValueError(f"Manifest needs columns: {required}")

# Calculate a 64-bit perceptual hash.
def perceptual_hash(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32))
    dct = cv2.dct(np.float32(resized))
    low = dct[:8, :8].flatten()
    bits = low > np.median(low[1:])

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value

def hamming(a, b):
    return (a ^ b).bit_count()

# Union-find: near-duplicate images stay in the same group.
parent = list(range(len(df)))
rank = [0] * len(df)

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    a, b = find(a), find(b)
    if a == b:
        return
    if rank[a] < rank[b]:
        a, b = b, a
    parent[b] = a
    if rank[a] == rank[b]:
        rank[a] += 1

print("Hashing images...")
hashes = []
for path in df["path"]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    hashes.append(perceptual_hash(p))

# BK-tree to find similar hashes efficiently.
class BKNode:
    def __init__(self, value, index):
        self.value = value
        self.indices = [index]
        self.children = {}

def query(node, value, radius, matches):
    if node is None:
        return
    distance = hamming(value, node.value)
    if distance <= radius:
        matches.extend((i, distance) for i in node.indices)

    low = max(0, distance - radius)
    high = distance + radius
    for edge, child in node.children.items():
        if low <= edge <= high:
            query(child, value, radius, matches)

def insert(node, value, index):
    if node is None:
        return BKNode(value, index)
    distance = hamming(value, node.value)
    if distance == 0:
        node.indices.append(index)
    else:
        node.children[distance] = insert(
            node.children.get(distance), value, index
        )
    return node

print("Grouping near-duplicates (Hamming distance <= 5)...")
root = None
edge_count = 0

for i, value in enumerate(hashes):
    matches = []
    query(root, value, HAMMING_LIMIT, matches)
    for j, distance in matches:
        if i != j:
            union(i, j)
            edge_count += 1
    root = insert(root, value, i)

df["group_id"] = [find(i) for i in range(len(df))]
df["group_id"] = pd.factorize(df["group_id"])[0]

group_label_counts = df.groupby("group_id")["label"].nunique()
conflict_groups = int((group_label_counts > 1).sum())

print("Images:", len(df))
print("Near-duplicate links:", edge_count)
print("Unique groups:", df["group_id"].nunique())
print("Groups with conflicting labels:", conflict_groups)

# Three partitions, keeping each group intact.
labels = df["label"].astype(str).to_numpy()
groups = df["group_id"].to_numpy()
indices = np.arange(len(df))

outer = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=SEED
)
remaining_idx, test_idx = next(
    outer.split(indices, labels, groups)
)

inner = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=SEED + 1
)
train_rel, val_rel = next(
    inner.split(
        remaining_idx,
        labels[remaining_idx],
        groups[remaining_idx],
    )
)

train_idx = remaining_idx[train_rel]
val_idx = remaining_idx[val_rel]

df["split"] = ""
df.loc[train_idx, "split"] = "train"
df.loc[val_idx, "split"] = "validation"
df.loc[test_idx, "split"] = "test"

# Verify no group crosses partitions.
assert df.groupby("group_id")["split"].nunique().max() == 1

# Verify each class exists in every partition.
coverage = pd.crosstab(df["label"], df["split"])
if (coverage == 0).any().any():
    raise RuntimeError(
        "A class is missing from a split. "
        "Do not train; inspect the group distribution first.\n"
        + coverage.to_string()
    )

# Keep group_id for auditability.
df.to_csv(OUTPUT, index=False)

print("\nImages per class and split:")
print(coverage.to_string())
print("\nImages per split:")
print(df["split"].value_counts().to_string())
print("\nSaved:", OUTPUT)
print("\nOriginal manifests and models were not overwritten.")
