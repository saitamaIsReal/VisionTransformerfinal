import os
from shapiq.interaction_values import InteractionValues

# Pfad zum Ordner mit .fsii-Dateien
fsii_dir = "output/results"

# Zielpfad für die Ausgabedatei
output_path = "top1_interaction_per_image.txt"

with open(output_path, "w", encoding="utf-8") as f_out:
    for file in sorted(os.listdir(fsii_dir)):
        if not file.endswith(".fsii"):
            continue

        fsii_path = os.path.join(fsii_dir, file)
        result = InteractionValues.load(fsii_path)

        # 2nd-Order-Werte extrahieren
        si = result.get_n_order(order=2)
        values_dict = si.dict_values

        # Top-1 nach Betrag
        top1 = max(values_dict.items(), key=lambda x: abs(x[1]))
        (i, j), value = top1

        # Schreiben
        f_out.write(f"{file}: Patch pair ({i}, {j}) → {value:.4f}\n")

print(f"[DONE] Top-1 2nd-Order Interactions gespeichert in: {output_path}")
