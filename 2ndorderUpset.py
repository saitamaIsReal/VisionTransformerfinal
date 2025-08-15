import os
from PIL import Image
from shapiq.interaction_values import InteractionValues
from shapiq.plot.upset import upset_plot
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# === SETTINGS ===
fsii_dir = "output/results"
output_dir = "output/upsets_2nd_only"
os.makedirs(output_dir, exist_ok=True)

def build_upset_plot_2ndonly(fsii_path, base_name):
    result = InteractionValues.load(fsii_path)
    second_order = result.get_n_order(order=2)

    # Top 10 nach Betrag
    top10_items = sorted(second_order.dict_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

    if not top10_items:
        print(f"[SKIPPED] Keine 2nd-order Werte in {base_name}")
        return

    # Feature-IDs extrahieren
    used_ids = sorted(set(i for pair, _ in top10_items for i in pair))
    id2new = {real: idx for idx, real in enumerate(used_ids)}
    remapped_pairs = [tuple(sorted((id2new[i], id2new[j]))) for (i, j) in [pair for pair, _ in top10_items]]
    values = [v for _, v in top10_items]

    subset = InteractionValues(
        values=np.array(values),
        index="sparse",
        n_players=len(used_ids),
        max_order=2,
        min_order=2,
        interaction_lookup={pair: idx for idx, pair in enumerate(remapped_pairs)},
        baseline_value=0.0
    )

    fig = upset_plot(subset, show=False, feature_names=[str(fid) for fid in used_ids])
    fig.suptitle(f"{base_name} – 2nd-Order Interactions", fontsize=12, weight='bold')
    fig.subplots_adjust(bottom=0.25)

    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    img = Image.frombuffer("RGBA", (w, h), canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
    plt_path = os.path.join(output_dir, f"{base_name}_2nd_only_upset.png")
    img.save(plt_path)
    print(f"[SAVED] {plt_path}")
    return img


# === LOOP ===
for file in os.listdir(fsii_dir):
    if not file.endswith(".fsii"):
        continue
    fsii_path = os.path.join(fsii_dir, file)
    base_name = file.replace(".fsii", "")
    build_upset_plot_2ndonly(fsii_path, base_name)

print("\n✅ [DONE] Alle 2nd-Order UpSet-Plots wurden erstellt und gespeichert.")
