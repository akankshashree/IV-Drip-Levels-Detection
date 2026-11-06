
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "results" / "temporal_mobilenetv2_test_predictions.csv"

df = pd.read_csv(FILE)

required = {"path", "label", "predicted_label"}
if not required.issubset(df.columns):
    raise ValueError(f"Missing columns: {required - set(df.columns)}")

# Extract timestamps from image filenames.
df["timestamp"] = pd.to_datetime(
    df["path"].str.extract(r"IMG(\d{14})")[0],
    format="%Y%m%d%H%M%S",
    errors="coerce",
)

df["correct"] = df["label"] == df["predicted_label"]

print("\n=== Overall test accuracy ===")
print(f"{df['correct'].mean() * 100:.2f}%")

print("\n=== Predictions by actual class ===")
print(pd.crosstab(df["label"], df["predicted_label"], margins=True))

print("\n=== Accuracy by chronological fifth within each class ===")
for label, group in df.groupby("label"):
    group = group.sort_values("timestamp").copy()
    group["time_bin"] = pd.qcut(
        group["timestamp"].rank(method="first"),
        q=5,
        labels=["first", "second", "middle", "fourth", "last"],
    )

    print(f"\n{label}")
    summary = group.groupby("time_bin", observed=True).agg(
        images=("correct", "size"),
        accuracy=("correct", "mean"),
        first_time=("timestamp", "min"),
        last_time=("timestamp", "max"),
    )
    summary["accuracy"] = (summary["accuracy"] * 100).round(2)
    print(summary.to_string())

# Find confidently wrong predictions, if class scores are available.
score_columns = [
    col for col in df.columns if col.startswith("score_")
]

if score_columns:
    df["predicted_score"] = df[score_columns].max(axis=1)
    wrong = df[~df["correct"]].sort_values(
        "predicted_score", ascending=False
    )

    print("\n=== 15 most confident incorrect predictions ===")
    print(
        wrong[
            ["timestamp", "label", "predicted_label", "predicted_score", "path"]
        ].head(15).to_string(index=False)
    )

print("\nAnalysis complete. No model or dataset files were changed.")
