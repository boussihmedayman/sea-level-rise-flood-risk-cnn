# -*- coding: utf-8 -*-
"""
PREDICT MAP - Sliding Window par pixel
Génère une carte de classes pixel par pixel via sliding window + stride,
avec filtres morphologiques par classe.
Tous les modèles sont des classifieurs binaires CNN (BinaryCNN).

Classes :
  0 = Inconnu
  1 = Riviere
  2 = Mer
  3 = Champs
  4 = Maisons
  5 = Foret
  6 = Usines

Usage :
  python src/predict_map.py --image ma_carte.png
  python src/predict_map.py --image ma_carte.png --stride 4 --output resultat.png --save-matrix
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import json
from scipy import ndimage
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

MODELS_DIR = Path("./models")
TILE_SIZE  = 16
STRIDE     = 16

CLASS_NAMES = {
    0: "Inconnu",
    1: "Riviere",
    2: "Mer",
    3: "Champs",
    4: "Maisons",
    5: "Foret",
    6: "Usines",
}

CLASS_COLORS = {
    0: (80,  80,  80,  160),
    1: (30,  144, 255, 160),
    2: (0,   105, 148, 160),
    3: (124, 205, 89,  160),
    4: (220, 80,  80,  160),
    5: (34,  139, 34,  160),
    6: (180, 100, 220, 160),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────
# MODELE
# ─────────────────────────────────────────

class BinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────
# AUTOENCODER (Foret & Champs)
# Architecture identique à train_autoencoder.py (ConvAutoencoder)
# ─────────────────────────────────────────

class AutoencoderCNN(nn.Module):
    """
    Autoencoder convolutionnel — copie exacte de ConvAutoencoder dans train_autoencoder.py.
    Entrée/Sortie : tuile RGB 16x16 (même TILE_SIZE que les CNN binaires).
    Encodeur : 16→8→4→2  (Conv stride=2 + BN)
    Décodeur : 2→4→8→16  (ConvTranspose stride=2 + BN)
    Score de détection = max(0, 1 - MSE / seuil).
    """
    def __init__(self):
        super().__init__()
        # Encodeur : 64x64 → 8x8
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # → 32x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → 16x16
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # →  8x8
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        # Décodeur : 8x8 → 64x64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # → 16x16
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # → 32x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),   # → 64x64
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        """Retourne l'erreur MSE pixel-level pour un batch."""
        x_hat = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=(1, 2, 3))  # (B,)


# Seuils lus depuis config_*.json si disponibles, sinon valeurs par défaut
AE_THRESHOLD_FORET  = 0.05
AE_THRESHOLD_CHAMPS = 0.05


# ─────────────────────────────────────────
# CHARGEMENT DES MODELES
# ─────────────────────────────────────────

def load_classifier(class_name, filename):
    model_path = MODELS_DIR / filename
    if not model_path.exists():
        print(f"  [SKIP] {class_name} introuvable : {model_path}")
        return None
    model = BinaryCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"  [OK] {class_name} (BinaryCNN)")
    return model


def load_autoencoder(class_name, filename):
    model_path = MODELS_DIR / filename
    if not model_path.exists():
        print(f"  [SKIP] {class_name} AE introuvable : {model_path}")
        return None
    model = AutoencoderCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"  [OK] {class_name} (AutoencoderCNN)")
    return model


# ─────────────────────────────────────────
# SLIDING WINDOW -> CARTES DE PROBABILITÉ
# ─────────────────────────────────────────

def build_prob_maps(img_arr, classifiers, autoencoders, ae_thresholds, stride):
    """
    Sliding window sur l'image entière.
    - classifiers   : dict {class_name -> BinaryCNN | None}  → prob via sigmoid(logit)
    - autoencoders  : dict {class_name -> AutoencoderCNN | None} → prob via 1 - erreur/seuil
    Retourne : dict {class_name -> np.array (H, W) probabilité moyenne par pixel}
    """
    h, w = img_arr.shape[:2]

    transform = transforms.Compose([
        transforms.Resize((TILE_SIZE, TILE_SIZE)),
        transforms.ToTensor(),
    ])

    all_models = {**{n: m for n, m in classifiers.items() if m is not None},
                  **{n: m for n, m in autoencoders.items() if m is not None}}

    prob_acc  = {name: np.zeros((h, w), dtype=np.float32) for name in all_models}
    count_map = np.zeros((h, w), dtype=np.float32)

    ae_names = set(autoencoders.keys())

    n_y = (h - TILE_SIZE) // stride + 1
    n_x = (w - TILE_SIZE) // stride + 1
    total = n_y * n_x
    done  = 0

    print(f"  Sliding window : {n_x}x{n_y} = {total} tuiles (stride={stride}px)")

    with torch.no_grad():
        for y in range(0, h - TILE_SIZE + 1, stride):
            for x in range(0, w - TILE_SIZE + 1, stride):

                tile_arr = img_arr[y:y+TILE_SIZE, x:x+TILE_SIZE]
                tile_t   = transform(Image.fromarray(tile_arr)).unsqueeze(0).to(DEVICE)

                # ── CNN binaires ─────────────────────────────────────
                for class_name, model in classifiers.items():
                    if model is None:
                        continue
                    logit = model(tile_t).item()
                    prob  = float(torch.sigmoid(torch.tensor(logit)))
                    prob_acc[class_name][y:y+TILE_SIZE, x:x+TILE_SIZE] += prob

                # ── Autoencoders ──────────────────────────────────────
                for class_name, model in autoencoders.items():
                    if model is None:
                        continue
                    err       = model.reconstruction_error(tile_t).item()
                    threshold = ae_thresholds.get(class_name, 0.05)
                    # Score ∈ [0, 1] : faible erreur → proche de 1
                    prob      = float(max(0.0, 1.0 - err / threshold))
                    prob_acc[class_name][y:y+TILE_SIZE, x:x+TILE_SIZE] += prob

                count_map[y:y+TILE_SIZE, x:x+TILE_SIZE] += 1.0

                done += 1
                if done % 5000 == 0 or done == total:
                    print(f"  Progression : {done}/{total} ({100*done/total:.0f}%)")

    count_map[count_map == 0] = 1.0
    prob_maps = {name: acc / count_map for name, acc in prob_acc.items()}
    return prob_maps


# ─────────────────────────────────────────
# CONSTRUCTION DE LA CARTE DE CLASSES
# ─────────────────────────────────────────

def build_class_map(prob_maps):
    """
    Pour chaque pixel, classe gagnante = CNN avec la probabilité la plus haute
    parmi ceux qui dépassent 0.5.
    En cas de conflit Champs/Foret : Champs gagne si prob_champs > prob_foret * 0.8
    """
    sample = next(iter(prob_maps.values()))
    H, W   = sample.shape

    best_score = np.full((H, W), -1.0, dtype=np.float32)
    class_map  = np.zeros((H, W), dtype=np.int32)

    name_to_idx = {v: k for k, v in CLASS_NAMES.items()}

    for class_name, prob_map in prob_maps.items():
        if prob_map is None:
            continue
        detected = prob_map > 0.5
        update   = detected & (prob_map > best_score)
        class_map[update]  = name_to_idx[class_name]
        best_score[update] = prob_map[update]

    # Champs et Foret sont maintenant des autoencoders :
    # la classe gagnante est simplement celle avec le score le plus haut (> 0.5)
    # Pas de priorité forcée — le best_score ci-dessus suffit.
    return class_map


# ─────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────

def build_overlay_image(img_pil, class_map):
    H, W = class_map.shape
    overlay     = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    overlay_arr = np.array(overlay)

    for cls_idx, (r, g, b, a) in CLASS_COLORS.items():
        mask = class_map == cls_idx
        overlay_arr[mask] = [r, g, b, a]

    overlay = Image.fromarray(overlay_arr, "RGBA")
    base    = img_pil.resize((W, H)).convert("RGBA")
    result  = Image.alpha_composite(base, overlay).convert("RGB")
    return result


def add_legend(image, present_classes):
    legend_h   = 30 + len(present_classes) * 22
    W, H       = image.size
    legend_img = Image.new("RGB", (W, H + legend_h), (30, 30, 30))
    legend_img.paste(image, (0, 0))

    draw = ImageDraw.Draw(legend_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    draw.text((10, H + 8), "Légende :", fill=(220, 220, 220), font=font)
    for i, cls_idx in enumerate(sorted(present_classes)):
        y = H + 28 + i * 22
        r, g, b, _ = CLASS_COLORS[cls_idx]
        draw.rectangle([10, y, 28, y + 16], fill=(r, g, b))
        draw.text((35, y), f"{cls_idx} - {CLASS_NAMES[cls_idx]}", fill=(220, 220, 220), font=font)

    return legend_img


# ─────────────────────────────────────────
# VOTE LOCAL POUR LES PIXELS INCONNUS
# ─────────────────────────────────────────

def local_vote_unknown(class_map, radius=24):
    result   = class_map.copy()
    max_iter = 5

    for iteration in range(max_iter):
        unknown_mask = (result == 0)
        n_unknown    = unknown_mask.sum()
        if n_unknown == 0:
            break

        print(f"  Vote local iter {iteration+1} : {n_unknown} pixels Inconnu...")

        new_map = result.copy()
        H, W    = result.shape
        changed = 0

        ys, xs = np.where(unknown_mask)
        for y, x in zip(ys, xs):
            y0 = max(0, y - radius)
            y1 = min(H, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(W, x + radius + 1)

            neighborhood = result[y0:y1, x0:x1]
            known        = neighborhood[neighborhood != 0]
            if len(known) == 0:
                continue

            counts    = np.bincount(known)
            new_map[y, x] = counts.argmax()
            changed += 1

        result = new_map
        print(f"    -> {changed} pixels réassignés")
        if changed == 0:
            break

    print(f"  Pixels Inconnu restants : {(result == 0).sum()}")
    return result


# ─────────────────────────────────────────
# STATISTIQUES
# ─────────────────────────────────────────

def print_stats(class_map):
    total = class_map.size
    print("\n── Statistiques ──────────────────────────────")
    unique, counts = np.unique(class_map, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        pct = 100 * count / total
        print(f"  {CLASS_NAMES.get(cls_idx, '?'):10s} (idx {cls_idx}) : {count:8d} px  ({pct:5.1f}%)")
    print(f"  {'TOTAL':10s}           : {total:8d} px")
    print("──────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(description="Prédiction carte de classes (pixel-level, CNN binaires + Autoencoders)")
    parser.add_argument("--image",       required=True)
    parser.add_argument("--stride",      type=int, default=STRIDE)
    parser.add_argument("--output",      default=None)
    parser.add_argument("--save-matrix", action="store_true")
    parser.add_argument("--no-vote",     action="store_true")
    parser.add_argument("--vote-radius", type=int, default=24)
    parser.add_argument("--no-legend",   action="store_true")

    parser.add_argument("--ae-threshold-foret",  type=float, default=AE_THRESHOLD_FORET,
                        help="Seuil MSE autoencoder Forêt (défaut=0.05)")
    parser.add_argument("--ae-threshold-champs", type=float, default=AE_THRESHOLD_CHAMPS,
                        help="Seuil MSE autoencoder Champs (défaut=0.05)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Erreur : image introuvable -> {image_path}")
        return

    output_path = Path(args.output) if args.output else image_path.with_name(image_path.stem + "_map.png")

    print(f"Device : {DEVICE}")
    print(f"Stride : {args.stride}px  |  Tile : {TILE_SIZE}px")
    print("\nChargement des modèles CNN binaires...")

    classifiers = {
        "Riviere": load_classifier("Riviere", "riviere_classifier_best.pth"),
        "Mer"    : load_classifier("Mer",     "mer_classifier_best.pth"),
        "Maisons": load_classifier("Maisons", "house_classifier_best.pth"),
        "Usines" : load_classifier("Usines",  "factory_classifier_best.pth"),
    }

    print("\nChargement des autoencoders (Foret & Champs)...")
    autoencoders = {
        "Foret" : load_autoencoder("Foret",  "autoencoder_Foret_best.pth"),
        "Champs": load_autoencoder("Champs", "autoencoder_Champs_best.pth"),
    }

    def load_ae_threshold(class_name, arg_value):
        """Lit le seuil depuis config_<Classe>.json si dispo, sinon utilise l'argument CLI."""
        cfg_path = MODELS_DIR / f"config_{class_name}.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            threshold = cfg.get("threshold", arg_value)
            print(f"  Seuil {class_name} lu depuis {cfg_path.name} : {threshold:.6f}")
            return threshold
        print(f"  Seuil {class_name} (défaut CLI) : {arg_value}")
        return arg_value

    ae_thresholds = {
        "Foret" : load_ae_threshold("Foret",  args.ae_threshold_foret),
        "Champs": load_ae_threshold("Champs", args.ae_threshold_champs),
    }
    print(f"  Seuils AE → Forêt={ae_thresholds['Foret']:.6f}  Champs={ae_thresholds['Champs']:.6f}")

    # ── Sliding window ────────────────────────────────────────────
    img     = Image.open(image_path).convert("RGB")
    img_arr = np.array(img)
    H, W    = img_arr.shape[:2]
    print(f"\nImage : {W}x{H}px")
    print("Sliding window en cours...")

    prob_maps = build_prob_maps(img_arr, classifiers, autoencoders, ae_thresholds, stride=args.stride)

    # ── Carte de classes brute ────────────────────────────────────
    print("\nConstruction de la carte de classes...")
    class_map = build_class_map(prob_maps)

    # ── Filtre taille Riviere ─────────────────────────────────────
    # <5000px -> supprimé
    # 5000-20000px -> gardé seulement si à moins de 90px d'une grande rivière (>=20000px)
    # >=20000px -> toujours gardé
    if classifiers.get("Riviere") is not None:
        riviere_idx = {v: k for k, v in CLASS_NAMES.items()}["Riviere"]
        inconnu_idx = 0
        rv_mask     = (class_map == riviere_idx).astype(np.uint8)
        labeled, n  = ndimage.label(rv_mask)
        sizes       = ndimage.sum(rv_mask, labeled, range(1, n + 1))

        # Masque des grandes rivières >= 20000px
        grande_mask = np.zeros_like(rv_mask)
        for i, size in enumerate(sizes):
            if size >= 20000:
                grande_mask[labeled == (i + 1)] = 1
        grande_dilated = ndimage.binary_dilation(grande_mask, iterations=90).astype(np.uint8)  # 90px

        removed_small  = 0
        removed_moyen  = 0
        for i, size in enumerate(sizes):
            region = labeled == (i + 1)
            if size < 5000:
                class_map[region] = inconnu_idx
                removed_small += 1
            elif size < 30000:
                if not grande_dilated[region].any():
                    class_map[region] = inconnu_idx
                    removed_moyen += 1

        print(f"\nFiltre Riviere : <5000px supprimées={removed_small} | 5000-20000px isolées supprimées={removed_moyen}")

    # ── Filtre taille minimale <400px → Inconnu (toutes classes sauf Rivière) ──
    def filtre_petites_zones(class_map, class_name, min_px=400, remplace_idx=0):
        idx  = {v: k for k, v in CLASS_NAMES.items()}[class_name]
        mask = (class_map == idx).astype(np.uint8)
        labeled, n = ndimage.label(mask)
        sizes      = ndimage.sum(mask, labeled, range(1, n + 1))
        removed    = 0
        for i, size in enumerate(sizes):
            if size < min_px:
                class_map[labeled == (i + 1)] = remplace_idx
                removed += 1
        remplace_nom = CLASS_NAMES.get(remplace_idx, "Inconnu")
        print(f"  Filtre {class_name} <{min_px}px : {removed} zones → {remplace_nom}")
        return class_map

    print("\nFiltres petites zones (<400px → Inconnu) :")
    for cls in ["Foret", "Champs", "Usines", "Mer", "Riviere"]:
        class_map = filtre_petites_zones(class_map, cls, min_px=400, remplace_idx=0)

    # ── Filtre côtier : Champs proche de la Mer → Rivière ────────
    # Si une grande zone Mer (>= 5000px) existe, les pixels Champs
    # situés à moins de 150px de cette Mer sont probablement de l'eau
    # côtière/sale → reclassés en Rivière.
    mer_idx     = {v: k for k, v in CLASS_NAMES.items()}["Mer"]
    riviere_idx = {v: k for k, v in CLASS_NAMES.items()}["Riviere"]
    champs_idx  = {v: k for k, v in CLASS_NAMES.items()}["Champs"]
    mer_mask    = (class_map == mer_idx).astype(np.uint8)
    labeled_mer, n_mer = ndimage.label(mer_mask)
    sizes_mer   = ndimage.sum(mer_mask, labeled_mer, range(1, n_mer + 1))

    grande_mer_mask = np.zeros_like(mer_mask)
    for i, size in enumerate(sizes_mer):
        if size >= 5000:
            grande_mer_mask[labeled_mer == (i + 1)] = 1

    if grande_mer_mask.any():
        # Zone forcée : < 400px → Rivière remplace Champs directement
        zone_forcee   = ndimage.binary_dilation(grande_mer_mask, iterations=400).astype(np.uint8)
        masque_force  = (class_map == champs_idx) & (zone_forcee == 1)
        n_force       = int(masque_force.sum())
        class_map[masque_force] = riviere_idx

        # Zone avantage : 400–500px → Rivière gagne si prob_riviere >= prob_champs * 0.7
        zone_elargie  = ndimage.binary_dilation(grande_mer_mask, iterations=500).astype(np.uint8)
        zone_avantage = (zone_elargie == 1) & (zone_forcee == 0)
        if "Riviere" in prob_maps and "Champs" in prob_maps:
            p_riv    = prob_maps["Riviere"]
            p_champs = prob_maps["Champs"]
            masque_avantage = (class_map == champs_idx) & zone_avantage & (p_riv >= p_champs * 0.5)
            n_avantage      = int(masque_avantage.sum())
            class_map[masque_avantage] = riviere_idx
        else:
            n_avantage = 0

        print(f"\nFiltre côtier : {n_force} px forcés → Rivière (<400px) | {n_avantage} px avantagés → Rivière (400-500px)")
    else:
        print("\nFiltre côtier : aucune grande zone Mer détectée, filtre ignoré")

    # ── Filtre grande Rivière : avantage sur Champs à 90px ───────
    # Si une zone Rivière >= 100000px existe, les pixels Champs dans
    # un rayon de 90px autour → Rivière si prob_riviere >= prob_champs * 0.7
    riv_mask    = (class_map == riviere_idx).astype(np.uint8)
    labeled_riv, n_riv = ndimage.label(riv_mask)
    sizes_riv   = ndimage.sum(riv_mask, labeled_riv, range(1, n_riv + 1))

    grande_riv_mask = np.zeros_like(riv_mask)
    for i, size in enumerate(sizes_riv):
        if size >= 100000:
            grande_riv_mask[labeled_riv == (i + 1)] = 1

    if grande_riv_mask.any():
        riv_dilated = ndimage.binary_dilation(grande_riv_mask, iterations=90).astype(np.uint8)
        if "Riviere" in prob_maps and "Champs" in prob_maps:
            p_riv    = prob_maps["Riviere"]
            p_champs = prob_maps["Champs"]

            # Labeliser tous les blocs Champs
            champs_mask_riv = (class_map == champs_idx).astype(np.uint8)
            labeled_ch, n_ch = ndimage.label(champs_mask_riv)
            sizes_ch = ndimage.sum(champs_mask_riv, labeled_ch, range(1, n_ch + 1))

            n_riv_avantage = 0
            for i in range(n_ch):
                bloc = labeled_ch == (i + 1)
                # Le bloc touche-t-il la zone des 90px autour de la grande rivière ?
                if not riv_dilated[bloc].any():
                    continue
                # Avantage : prob_riviere >= prob_champs * 0.7 en moyenne sur le bloc
                if p_riv[bloc].mean() >= p_champs[bloc].mean() * 0.4:
                    class_map[bloc] = riviere_idx
                    n_riv_avantage += int(bloc.sum())
        else:
            n_riv_avantage = 0
        print(f"\nFiltre grande Rivière : {n_riv_avantage} px Champs → Rivière (blocs touchant les 90px autour des zones ≥100000px)")
    else:
        print("\nFiltre grande Rivière : aucune zone ≥100000px, filtre ignoré")

    # ── Filtre enclave : petites zones dans un gros bloc Rivière/Mer ──
    # Si une zone d'une autre classe est entièrement contenue dans
    # un grand bloc Rivière ou Mer, et fait < 30000px → reclassée dans
    # cette classe d'eau (Rivière ou Mer), puisqu'elle en est cernée.
    for eau_class in ["Riviere", "Mer"]:
        eau_idx  = {v: k for k, v in CLASS_NAMES.items()}[eau_class]
        eau_mask = (class_map == eau_idx).astype(np.uint8)
        labeled_eau, n_eau = ndimage.label(eau_mask)
        sizes_eau = ndimage.sum(eau_mask, labeled_eau, range(1, n_eau + 1))

        # Construire un masque des "grands blocs" eau
        grands_blocs_eau = np.zeros_like(eau_mask)
        for i, size in enumerate(sizes_eau):
            if size >= 100000:
                grands_blocs_eau[labeled_eau == (i + 1)] = 1

        if not grands_blocs_eau.any():
            continue

        # Trouver les zones enclavées via flood-fill depuis les bords
        # (fonctionne même si la zone touche le bord de l'image)
        non_eau = (grands_blocs_eau == 0).astype(np.uint8)
        # Seed = tous les pixels non-eau connectés au bord
        seed = np.zeros_like(non_eau)
        seed[0, :]  = non_eau[0, :]
        seed[-1, :] = non_eau[-1, :]
        seed[:, 0]  = non_eau[:, 0]
        seed[:, -1] = non_eau[:, -1]
        # Propagation : tout ce qui est atteignable depuis le bord sans traverser l'eau
        exterieur = ndimage.binary_propagation(seed.astype(bool), mask=non_eau.astype(bool))
        zone_enclave = non_eau.astype(bool) & ~exterieur

        # Labeliser les zones enclavées
        labeled_enc, n_enc = ndimage.label(zone_enclave)
        sizes_enc = ndimage.sum(zone_enclave.astype(np.uint8), labeled_enc, range(1, n_enc + 1))
        removed_enc = 0
        for i, size in enumerate(sizes_enc):
            if size < 30000:
                class_map[labeled_enc == (i + 1)] = eau_idx
                removed_enc += 1

        # Presqu'îles à cou étroit : reliées au continent par un fin passage de
        # terre, donc invisibles pour le flood-fill ci-dessus (qui ne détecte que
        # les îlots totalement déconnectés). On érode le masque non-eau pour
        # casser les cous étroits (<2*COU_RADIUS px), puis on redilate les
        # fragments ainsi isolés pour retrouver leur forme d'origine.
        COU_RADIUS = 5  # casse les ponts de terre plus étroits qu'environ 10px
        non_eau_bool = non_eau.astype(bool)
        eroded = ndimage.binary_erosion(non_eau_bool, iterations=COU_RADIUS)
        labeled_er, n_er = ndimage.label(eroded)
        removed_presqu = 0
        if n_er > 0:
            sizes_er = ndimage.sum(eroded.astype(np.uint8), labeled_er, range(1, n_er + 1))
            continent_label = int(np.argmax(sizes_er)) + 1  # le plus gros fragment = le continent
            for i, size in enumerate(sizes_er):
                label_id = i + 1
                if label_id == continent_label or size == 0:
                    continue
                # Redilater pour récupérer la forme d'origine de la presqu'île
                morceau = ndimage.binary_dilation(labeled_er == label_id, iterations=COU_RADIUS)
                morceau = morceau & non_eau_bool & (class_map != eau_idx)
                taille_morceau = int(morceau.sum())
                if 0 < taille_morceau < 30000:
                    class_map[morceau] = eau_idx
                    removed_presqu += taille_morceau

        print(f"\nFiltre enclave {eau_class} : {removed_enc} zones <30000px → {eau_class} "
              f"| presqu'îles cernées d'eau : {removed_presqu}px → {eau_class}")

    # ── Filtre petites zones (2e passage) ─────────────────────────
    # Les filtres côtier, grande Rivière et enclave reclassent des pixels
    # (Champs→Rivière, enclave→eau, etc.) et peuvent créer de nouveaux petits
    # fragments <400px qui n'ont jamais été vus par le filtre du début.
    # On refait donc passer le même filtre ici, juste avant le vote local,
    # pour nettoyer ces résidus (ils seront ensuite comblés par le vote).
    print("\nFiltres petites zones (2e passage, <400px → Inconnu) :")
    for cls in ["Foret", "Champs", "Usines", "Mer", "Riviere"]:
        class_map = filtre_petites_zones(class_map, cls, min_px=400, remplace_idx=0)

    # ── Vote local pixels Inconnu ─────────────────────────────────
    if not args.no_vote:
        print("\nVote local pour les pixels Inconnu...")
        class_map = local_vote_unknown(class_map, radius=args.vote_radius)

    # ── Stats ─────────────────────────────────────────────────────
    print_stats(class_map)

    # ── Image résultat ────────────────────────────────────────────
    result_image    = build_overlay_image(img, class_map)
    present_classes = set(np.unique(class_map).tolist())
    if not args.no_legend:
        result_image = add_legend(result_image, present_classes)

    result_image.save(output_path)
    print(f"\nImage de détection sauvegardée : {output_path}")

    if args.save_matrix:
        npy_path = output_path.with_suffix(".npy")
        csv_path = output_path.with_suffix(".csv")
        np.save(npy_path, class_map)
        np.savetxt(csv_path, class_map, delimiter=",", fmt="%d")
        print(f"Matrice NumPy : {npy_path}")
        print(f"Matrice CSV   : {csv_path}")

    print(f"\nMatrice : shape={class_map.shape}  dtype={class_map.dtype}")
    return class_map, result_image


if __name__ == "__main__":
    main()