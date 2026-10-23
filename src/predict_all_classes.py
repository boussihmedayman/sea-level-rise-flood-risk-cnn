# -*- coding: utf-8 -*-
"""
PREDICT ALL CLASSES
A partir d'UNE image, genere 7 images en sortie :
  - 6 images de detection individuelle, une par classe
    (Riviere, Mer, Champs, Maisons, Foret, Usines)
  - 1 image "carte complete" avec les 6 classes superposees,
    en relancant directement predict_map.py (donc AVEC ses filtres
    morphologiques, exactement le resultat obtenu en
    lancant predict_map.py directement).

Les 6 images individuelles reutilisent le meme sliding window +
les memes modeles (BinaryCNN pour Riviere/Mer/Maisons/Usines,
AutoencoderCNN pour Foret/Champs) que predict_map.py, calcules en
UNE seule passe (pas 6 passes separees), donc rapide. Pas de filtres
morphologiques sur ces 6 images : c'est la detection brute de chaque
modele, pixel par pixel.

IMPORTANT : ce script doit etre lance depuis le meme dossier que
predict_map.py (il l'importe comme module et le relance en sous-processus
pour la carte finale).

Usage :
  python src/predict_all_classes.py --image tuile.png
  python src/predict_all_classes.py --image tuile.png --stride 8 --output-dir resultats
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import predict_map  # reutilise directement les modeles / la logique de predict_map.py


# ─────────────────────────────────────────
# OVERLAY POUR UNE SEULE CLASSE
# ─────────────────────────────────────────

def build_single_class_overlay(img_arr, mask, color, alpha=0.55):
    """Applique un overlay colore (blend) sur les pixels detectes, garde le reste intact."""
    overlay = img_arr.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask] = overlay[mask] * (1 - alpha) + color_arr * alpha
    return overlay.astype(np.uint8)


def add_title(img_pil, text):
    """Ajoute une petite bande de titre en haut de l'image, comme predict_map.add_legend."""
    W, H = img_pil.size
    band_h = 34
    out = Image.new("RGB", (W, H + band_h), (30, 30, 30))
    out.paste(img_pil, (0, band_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 8), text, fill=(230, 230, 230), font=font)
    return out


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Genere les 6 detections individuelles + la carte complete (predict_map.py)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--models-dir", default="./models")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output-dir", default=None, help="Dossier de sortie (defaut : meme dossier que l'image)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil de detection par classe (defaut 0.5, comme predict_map.py)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Erreur : image introuvable -> {image_path}")
        return

    out_dir = Path(args.output_dir) if args.output_dir else image_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    predict_map.MODELS_DIR = Path(args.models_dir)

    # ── Chargement des 6 modeles (comme predict_map.py) ─────────────
    print("Chargement des modeles CNN binaires...")
    classifiers = {
        "Riviere": predict_map.load_classifier("Riviere", "riviere_classifier_best.pth"),
        "Mer"    : predict_map.load_classifier("Mer",     "mer_classifier_best.pth"),
        "Maisons": predict_map.load_classifier("Maisons", "house_classifier_best.pth"),
        "Usines" : predict_map.load_classifier("Usines",  "factory_classifier_best.pth"),
    }
    print("Chargement des autoencoders (Foret & Champs)...")
    autoencoders = {
        "Foret" : predict_map.load_autoencoder("Foret",  "autoencoder_Foret_best.pth"),
        "Champs": predict_map.load_autoencoder("Champs", "autoencoder_Champs_best.pth"),
    }

    def load_ae_threshold(class_name, default):
        cfg_path = predict_map.MODELS_DIR / f"config_{class_name}.json"
        if cfg_path.exists():
            import json
            with open(cfg_path) as f:
                cfg = json.load(f)
            return cfg.get("threshold", default)
        return default

    ae_thresholds = {
        "Foret" : load_ae_threshold("Foret",  predict_map.AE_THRESHOLD_FORET),
        "Champs": load_ae_threshold("Champs", predict_map.AE_THRESHOLD_CHAMPS),
    }

    # ── Une seule passe de sliding window pour les 6 modeles ────────
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img)
    print(f"\nImage : {img_arr.shape[1]}x{img_arr.shape[0]}px")
    print("Sliding window (6 modeles en une passe)...")
    prob_maps = predict_map.build_prob_maps(img_arr, classifiers, autoencoders, ae_thresholds, stride=args.stride)

    # ── 1 image de detection par classe ──────────────────────────────
    print("\nGeneration des 6 images individuelles...")
    for class_name, prob_map in prob_maps.items():
        mask = prob_map > args.threshold
        pct  = round(float(mask.mean() * 100), 1)
        color = predict_map.CLASS_COLORS[{v: k for k, v in predict_map.CLASS_NAMES.items()}[class_name]][:3]

        overlay_arr = build_single_class_overlay(img_arr, mask, color)
        overlay_img = Image.fromarray(overlay_arr, "RGB")
        overlay_img = add_title(overlay_img, f"{class_name} - {pct}% detecte")

        out_path = out_dir / f"{stem}_{class_name}.png"
        overlay_img.save(out_path)
        print(f"  [OK] {class_name:10s} ({pct:5.1f}%) -> {out_path}")

    # ── 7e image : carte complete via predict_map.py (avec filtres) ─
    print("\nGeneration de la carte complete (predict_map.py, avec filtres)...")
    map_out_path = out_dir / f"{stem}_map.png"
    cmd = [
        sys.executable, str(Path(__file__).parent / "predict_map.py"),
        "--image", str(image_path),
        "--output", str(map_out_path),
        "--stride", str(args.stride),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:])  # dernieres lignes du log de predict_map.py
    if result.returncode != 0:
        print("Erreur lors de l'appel a predict_map.py :")
        print(result.stderr[-2000:])
        return
    print(f"  [OK] Carte complete -> {map_out_path}")

    print("\nTermine : 7 images generees dans", out_dir)


if __name__ == "__main__":
    main()
