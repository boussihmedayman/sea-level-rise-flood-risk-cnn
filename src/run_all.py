# -*- coding: utf-8 -*-
"""
RUN ALL — Prédiction de classes (predict_map.py) sur les tuiles validées
=========================================================================
Ne traite que les tuiles dont le status est "valide" dans l'index produit
par eec_flood_tiles_extractor.py (./outputs/eec_flood_tiles/tiles_flood_index.json).
Les tuiles écartées ("mer_permanente", "hors_risque") sont ignorées.

Pour chaque tuile validée :
  1. Lance predict_map.py → carte de classes colorée
       results/Tuile{id}_map.png
  2. Construit une comparaison côte à côte [Tuile originale | Tuile colorée par classes]
       results/comparaisons/tile_{id:03d}_comparaison.png

Reprise automatique via results/progress.json (clé = nom du fichier image).

Usage :
  python src/run_all.py   (depuis la racine du dépôt)
"""

from pathlib import Path
import subprocess
import sys
import json

from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

INPUT_DIR     = Path("Tuiles")
OUTPUT_DIR    = Path("results")
COMP_DIR      = OUTPUT_DIR / "comparaisons"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Index produit par eec_flood_tiles_extractor.py (ne contient déjà que les
# tuiles non écartées, mais on revérifie le status explicitement par sécurité
# si jamais tiles_flood_index_full.json est utilisé à la place un jour)
INDEX_PATH    = Path("./outputs/eec_flood_tiles/tiles_flood_index.json")

# Extensions acceptées pour les images sources (même ordre de priorité
# que find_satellite_image dans eec_flood_visualizer.py)
IMG_EXTS      = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

OUTPUT_DIR.mkdir(exist_ok=True)
COMP_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────

def find_tile_image(tile_id: int) -> Path | None:
    """Cherche Tuiles/Tuile{id}.ext parmi les extensions acceptées."""
    for ext in IMG_EXTS:
        p = INPUT_DIR / f"Tuile{tile_id}{ext}"
        if p.exists():
            return p
    return None


def make_comparison(original_path: Path, class_map_path: Path, tile_id: int) -> Image.Image:
    """[Tuile originale] | [Tuile colorée par classes (sortie predict_map.py)]"""
    original  = Image.open(original_path).convert("RGB")
    class_img = Image.open(class_map_path).convert("RGB")

    if class_img.size != original.size:
        class_img = class_img.resize(original.size, Image.LANCZOS)

    W, H    = original.size
    SEP     = 6
    HEADER  = 50
    canvas  = Image.new("RGB", (W * 2 + SEP, H + HEADER), (20, 20, 20))

    draw   = ImageDraw.Draw(canvas)
    font_t = ImageFont.load_default(size=30)
    title  = f"Tuile {tile_id}  —  Originale | Classes prédites"
    tb     = draw.textbbox((0, 0), title, font=font_t)
    tx     = (W * 2 + SEP - (tb[2] - tb[0])) // 2
    ty     = (HEADER - (tb[3] - tb[1])) // 2
    draw.text((tx, ty), title, fill=(220, 220, 220), font=font_t)

    canvas.paste(original, (0, HEADER))
    draw.rectangle([W, HEADER, W + SEP, H + HEADER], fill=(60, 60, 60))
    canvas.paste(class_img, (W + SEP, HEADER))

    return canvas


# ─────────────────────────────────────────────────────────────────
# CHARGEMENT DES TUILES VALIDÉES
# ─────────────────────────────────────────────────────────────────

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"Index introuvable : {INDEX_PATH}\n"
        "Lance d'abord eec_flood_tiles_extractor.py"
    )

with open(INDEX_PATH) as f:
    index = json.load(f)

valid_ids = sorted(e["id"] for e in index if e.get("status", "valide") == "valide")
print(f"{len(valid_ids)} tuiles validées dans l'index ({INDEX_PATH})")

# Charger la progression existante
if PROGRESS_FILE.exists():
    with open(PROGRESS_FILE) as f:
        done = set(json.load(f))
    print(f"Progression chargée : {len(done)} tuiles déjà traitées")
else:
    done = set()

n_missing = 0

for tid in valid_ids:
    img = find_tile_image(tid)
    if img is None:
        print(f"  [!] Tuile {tid:03d} — image source introuvable dans {INPUT_DIR}/")
        n_missing += 1
        continue

    if img.name in done:
        print(f"  [SKIP] {img.name}")
        continue

    output_png = OUTPUT_DIR / f"{img.stem}_map.png"
    print(f"\nTraitement : {img.name}  (tuile {tid:03d})")

    try:
        subprocess.run([
            sys.executable,
            str(Path(__file__).parent / "predict_map.py"),
            "--image", str(img),
            "--stride", "4",
            "--save-matrix",
            "--output", str(output_png)
        ], check=True)

        # Comparaison côte à côte [Tuile originale | Tuile colorée par classes]
        comparison = make_comparison(img, output_png, tid)
        comp_path  = COMP_DIR / f"tile_{tid:03d}_comparaison.png"
        comparison.save(comp_path, optimize=True)
        print(f"  [✓] Comparaison → {comp_path.name}")

        # Marquer comme traitée
        done.add(img.name)
        with open(PROGRESS_FILE, "w") as f:
            json.dump(list(done), f)

    except subprocess.CalledProcessError as e:
        print(f"  [ERREUR] {img.name} : {e}")
        print(f"  La tuile sera retentée au prochain lancement")

print(f"\nTerminé. {len(done)}/{len(valid_ids)} tuiles validées traitées."
      + (f"  ({n_missing} images sources manquantes)" if n_missing else ""))