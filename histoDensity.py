import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapiq.interaction_values import InteractionValues
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

fsii_dir = "output/results"
output_dir = "output/histograms_density"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(fsii_dir):
    if not file.endswith(".fsii"):
        continue

    path = os.path.join(fsii_dir, file)
    print(f"[INFO] Verarbeite: {file}")
    result = InteractionValues.load(path)

    base_filename = file.replace(".fsii", "")

    # ===== ORDER 1 =====
    order1 = result.get_n_order(order=1)
    values_1 = np.abs([v for v in order1.dict_values.values()])
    fig1 = plt.figure(figsize=(6, 4))
    sns.histplot(values_1, bins=30, stat="density", color="darkorange")
    plt.title(f"Dichte – Shapley-Werte\n({base_filename})", fontsize=10)
    plt.xlabel("Wert (absolut)")
    plt.ylabel("Dichte")
    plt.tight_layout()
    canvas1 = FigureCanvas(fig1)
    canvas1.draw()
    w, h = canvas1.get_width_height()
    Image.frombuffer("RGBA", (w, h), canvas1.buffer_rgba(), "raw", "RGBA", 0, 1)\
         .save(os.path.join(output_dir, f"{base_filename}_shapley_only_density.png"))
    plt.close(fig1)

    # ===== ORDER 2 =====
    order2 = result.get_n_order(order=2)
    values_2 = np.abs([v for v in order2.dict_values.values()])
    fig2 = plt.figure(figsize=(6, 4))
    sns.histplot(values_2, bins=30, stat="density", color="purple")
    plt.title(f"Dichte – 2nd-Order Interactions\n({base_filename})", fontsize=10)
    plt.xlabel("Wert (absolut)")
    plt.ylabel("Dichte")
    plt.tight_layout()
    canvas2 = FigureCanvas(fig2)
    canvas2.draw()
    w, h = canvas2.get_width_height()
    Image.frombuffer("RGBA", (w, h), canvas2.buffer_rgba(), "raw", "RGBA", 0, 1)\
         .save(os.path.join(output_dir, f"{base_filename}_2nd_only_density.png"))
    plt.close(fig2)

    # ===== ORDER 1 + 2 =====
    values_all = np.abs([v for k, v in result.dict_values.items() if len(k) in [1, 2]])
    fig3 = plt.figure(figsize=(6, 4))
    sns.histplot(values_all, bins=30, stat="density", color="royalblue")
    plt.title(f"Dichte – Shapley + Interaktionen\n({base_filename})", fontsize=10)
    plt.xlabel("Wert (absolut)")
    plt.ylabel("Dichte")
    plt.tight_layout()
    canvas3 = FigureCanvas(fig3)
    canvas3.draw()
    w, h = canvas3.get_width_height()
    Image.frombuffer("RGBA", (w, h), canvas3.buffer_rgba(), "raw", "RGBA", 0, 1)\
         .save(os.path.join(output_dir, f"{base_filename}_combined_1_2_density.png"))
    plt.close(fig3)

print("\n✅ [DONE] Alle Histogramme mit density=True gespeichert.")
