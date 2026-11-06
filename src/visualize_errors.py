
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "temporal_mobilenetv2_test_predictions.csv"
OUT = ROOT / "results" / "temporal_error_contact_sheet.png"

df = pd.read_csv(CSV)
df = df[df["label"] != df["predicted_label"]].copy()

score_cols = [c for c in df.columns if c.startswith("score_")]
if not score_cols:
    raise ValueError("No score columns found in prediction CSV.")

df["confidence"] = df[score_cols].max(axis=1)
df = df.sort_values("confidence", ascending=False).head(12)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for ax, (_, row) in zip(axes, df.iterrows()):
    path = Path(row["path"])
    try:
        with Image.open(path) as im:
            ax.imshow(im.convert("RGB").resize((350, 350)))
        ax.set_title(
            f"True: {row['label']}\n"
            f"Pred: {row['predicted_label']} "
            f"({row['confidence']:.1%})",
            fontsize=9,
        )
    except Exception as exc:
        ax.text(0.5, 0.5, f"Could not load image\n{exc}", ha="center")
    ax.axis("off")

for ax in axes[len(df):]:
    ax.axis("off")

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)

print("Contact sheet saved:", OUT)
