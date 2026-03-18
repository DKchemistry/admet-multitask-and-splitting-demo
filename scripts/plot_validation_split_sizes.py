from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Validation fold sizes from the logs / split logic
butina_vals = [
    748,
    749,
    747,
    749,
    747,
    747,
    751,
    747,
    746,
    749,
    748,
    748,
    749,
    748,
    747,
    742,
    741,
    772,
    741,
    744,
    746,
    750,
    746,
    751,
    747,
]

scaffold_vals = [
    748,
    749,
    747,
    749,
    747,
    748,
    747,
    749,
    747,
    749,
    746,
    745,
    758,
    746,
    745,
    747,
    747,
    747,
    750,
    749,
    749,
    748,
    748,
    747,
    748,
]

# N = 3740, n_folds = 5, so every random validation fold has exactly 748 rows
random_vals = [748] * 25

df = pd.DataFrame(
    {
        "method": (
            ["Random"] * len(random_vals)
            + ["Scaffold"] * len(scaffold_vals)
            + ["Butina"] * len(butina_vals)
        ),
        "val_rows": random_vals + scaffold_vals + butina_vals,
    }
)

order = ["Random", "Scaffold", "Butina"]

fig, ax = plt.subplots(figsize=(7.6, 4.8))

groups = [df[df["method"] == method]["val_rows"] for method in order]
ax.boxplot(groups, tick_labels=order, widths=0.5)

# Overlay raw points with small deterministic horizontal offsets
for i, method in enumerate(order, start=1):
    vals = df[df["method"] == method]["val_rows"].tolist()
    offsets = [((j % 7) - 3) * 0.03 for j in range(len(vals))]
    xs = [i + off for off in offsets]
    ax.scatter(xs, vals, s=28)

# Ideal 20% line
ax.axhline(3740 / 5, linestyle="--", linewidth=1)

ax.set_ylabel("Validation rows")
ax.set_title("Validation fold sizes pooled across repeats")
ax.grid(True, axis="y", alpha=0.3)

project_root = Path(__file__).resolve().parent.parent
out_path = project_root / "figures" / "validation_fold_sizes.png"
out_path.parent.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
