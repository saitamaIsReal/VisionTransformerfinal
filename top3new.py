import os
from PIL import Image, ImageDraw, ImageFont
from shapiq.interaction_values import InteractionValues

# === SETTINGS ===
fsii_dir = "output/results"
image_dir = "images"
output_dir = "output/top_interactions_rebased"
os.makedirs(output_dir, exist_ok=True)

# === Modellkonfigurationen: Bildgröße + Patchgröße
model_configs = {
    "google_vit-base-patch32-384": {"img_size": 384, "patch_size": 32},
    "facebook_deit-tiny-patch16-224": {"img_size": 224, "patch_size": 16},
    "akahana_vit-base-cats-vs-dogs": {"img_size": 224, "patch_size": 16},
}

# === Index -> Dateiname
index_to_filename = {
    "0": "cat1.jpg",
    "1": "cat2.jpg",
    "2": "cat3.jpg",
    "3": "cat4.jpg",
    "4": "lucky.jpeg",    
    "5": "dog2.jpg",
    "6": "dog3.jpg",
    "7": "dog4.jpg",
}

# === Farben für Interaktionen
colors = {
    "mixed": (255, 0, 0),     # rot
    "pos":   (0, 200, 0),     # grün
    "neg":   (0, 100, 255),   # blau
}

def draw_patch_outline(img, patch_id, color, n_patches_per_row, cell):
    draw = ImageDraw.Draw(img)
    r, c = divmod(patch_id, n_patches_per_row)
    x1 = c * cell
    y1 = r * cell
    x2 = x1 + cell
    y2 = y1 + cell
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

def draw_grid(img: Image.Image, n_patches_per_row: int, cell: int):
    draw = ImageDraw.Draw(img)
    for r in range(n_patches_per_row):
        for c in range(n_patches_per_row):
            x1, y1 = c*cell, r*cell
            x2, y2 = x1+cell, y1+cell
            draw.rectangle([x1, y1, x2, y2], outline="gray", width=1)
            idx = r*n_patches_per_row + c
            try:
                font = ImageFont.truetype("arial.ttf", size=7)
            except:
                font = ImageFont.load_default()
            draw.text((x1+2, y1+2), str(idx), fill="gray", font=font)

def overlay_top3_interactions(img, fsii_path, output_path, tag, patch_size):
    result = InteractionValues.load(fsii_path)
    second_order = result.get_n_order(order=2)

    image_size = img.size[0]
    n_patches_per_row = image_size // patch_size
    cell = patch_size

    items = sorted(second_order.dict_values.items(), key=lambda x: abs(x[1]), reverse=True)
    if tag == "pos":
        items = [item for item in items if item[1] > 0][:3]
    elif tag == "neg":
        items = [item for item in items if item[1] < 0][:3]
    else:
        items = items[:3]

    overlay = img.copy()
    draw_grid(overlay, n_patches_per_row, cell)
    for (i, j), _ in items:
        draw_patch_outline(overlay, i, colors[tag], n_patches_per_row, cell)
        draw_patch_outline(overlay, j, colors[tag], n_patches_per_row, cell)

    overlay.save(output_path)
    print(f"[SAVED] {output_path}")

# === Haupt-Loop
for file in os.listdir(fsii_dir):
    if not file.endswith(".fsii"):
        continue

    fsii_path = os.path.join(fsii_dir, file)
    base = file.replace(".fsii", "")
    img_idx, model_name_raw = base.split("_", 1)

    filename = index_to_filename.get(img_idx, None)
    if filename is None:
        print(f"[SKIPPED] Kein Mapping für Index {img_idx}")
        continue

    image_path = os.path.join(image_dir, filename)
    if not os.path.exists(image_path):
        print(f"[SKIPPED] Datei fehlt: {image_path}")
        continue

    # Modellinfos laden
    model_conf = model_configs[model_name_raw]
    model_img_size = model_conf["img_size"]
    patch_size = model_conf["patch_size"]

    # Resize wie im Hauptcode
    img_raw = Image.open(image_path).convert("RGB")
    img_processed = img_raw.resize((model_img_size, model_img_size))

    # Speichern
    output_base = os.path.join(output_dir, f"{img_idx}_{model_name_raw}")
    resized_img_path = f"{output_base}_resized.png"
    img_processed.save(resized_img_path)

    # Top 3 Interaktionen (mixed pos neg)
    for tag in ["mixed", "pos", "neg"]:
        out_path = f"{output_base}_top3_{tag}.png"
        overlay_top3_interactions(img_processed, fsii_path, out_path, tag, patch_size)
