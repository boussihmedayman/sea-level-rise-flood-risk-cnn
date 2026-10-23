# -*- coding: utf-8 -*-
"""
PREDICTION CLASSIFIEUR BINAIRE (CNN)
Version generalisee de predict_house_classifier.py : au lieu d'etre
fige sur "Maisons", on choisit la classe avec --class parmi les 4
classes gerees par un CNN binaire (BinaryCNN, meme architecture que
dans predict_map.py) :

    Riviere, Mer, Maisons, Usines

NB : Foret et Champs ne sont PAS gerees ici, elles utilisent une
architecture differente (autoencoder one-class) -> utilise
predict_autoencoder.py --class Foret / --class Champs pour celles-la.

Usage :
  python src/predict_classifier.py --class Maisons --image mon_image.png
  python src/predict_classifier.py --class Riviere --image mon_image.png --threshold 0.6
"""

import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
import argparse

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODELS_DIR = Path("./models")

# Fichier de poids pour chaque classe geree par un CNN binaire
# (memes noms de fichiers que dans predict_map.py)
CLASS_MODEL_FILES = {
    "Riviere": "riviere_classifier_best.pth",
    "Mer"    : "mer_classifier_best.pth",
    "Maisons": "house_classifier_best.pth",
    "Usines" : "factory_classifier_best.pth",
}

# Couleur d'overlay par classe (R, G, B), coherente avec predict_map.py
CLASS_COLORS = {
    "Riviere": (30,  144, 255),
    "Mer"    : (0,   105, 148),
    "Maisons": (220, 80,  80),
    "Usines" : (180, 100, 220),
}

IMG_SIZE  = 16
STRIDE    = 4
THRESHOLD = 0.5  # aligne avec le seuil de decision de predict_map.py

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------
# CNN (architecture identique a BinaryCNN dans predict_map.py)
# --------------------------------------------------

class BinaryCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model(class_name):
    model_path = MODELS_DIR / CLASS_MODEL_FILES[class_name]

    if not model_path.exists():
        raise FileNotFoundError(f"Modele non trouve : {model_path}")

    model = BinaryCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print("Modele charge :", model_path)

    return model

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_image(model, image_path, threshold):

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img)
    h, w = img_arr.shape[:2]

    prob_map  = np.zeros((h, w), dtype=float)
    count_map = np.zeros((h, w), dtype=float)

    total_tiles = (
        ((h - IMG_SIZE) // STRIDE + 1) *
        ((w - IMG_SIZE) // STRIDE + 1)
    )

    print("Image :", w, "x", h)
    print("Tuiles :", total_tiles)

    with torch.no_grad():
        for y in range(0, h - IMG_SIZE + 1, STRIDE):
            for x in range(0, w - IMG_SIZE + 1, STRIDE):

                tile = img_arr[y:y+IMG_SIZE, x:x+IMG_SIZE]
                tile_t = transform(Image.fromarray(tile)).unsqueeze(0).to(DEVICE)

                logits = model(tile_t)
                prob = float(torch.sigmoid(logits).cpu().numpy()[0][0])

                prob_map[y:y+IMG_SIZE, x:x+IMG_SIZE]  += prob
                count_map[y:y+IMG_SIZE, x:x+IMG_SIZE] += 1

    count_map[count_map == 0] = 1
    prob_map /= count_map

    detected_mask = prob_map > threshold
    pct = round(float(detected_mask.mean() * 100), 1)

    print("Detecte :", pct, "%")

    return prob_map, detected_mask, pct

# --------------------------------------------------
# VISUALISATION
# --------------------------------------------------

def visualize(img_path, prob_map, detected_mask, pct, class_name):

    img = np.array(Image.open(img_path).convert("RGB"))
    stem = Path(img_path).stem
    r, g, b = CLASS_COLORS[class_name]

    overlay = img.copy()
    overlay[detected_mask, 0] = r
    overlay[detected_mask, 1] = g
    overlay[detected_mask, 2] = b

    # ------------------------------
    # Comparaison
    # ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    p1 = mpatches.Patch(color=np.array([r, g, b]) / 255, label=class_name)
    axes[1].legend(handles=[p1])
    axes[1].set_title(f"{class_name} detecte ({pct}%)")
    axes[1].axis("off")

    plt.tight_layout()
    out1 = f"{stem}_{class_name}_prediction.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print("Sauvegarde :", out1)

    # ------------------------------
    # Heatmap
    # ------------------------------
    plt.figure(figsize=(10, 6))
    plt.imshow(prob_map, cmap="hot")
    plt.colorbar(label=f"Probabilite {class_name}")
    plt.title(f"Heatmap probabilite {class_name}")
    plt.axis("off")

    out2 = f"{stem}_{class_name}_heatmap.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()
    print("Sauvegarde :", out2)

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    global MODELS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--class", dest="class_name", required=True,
        choices=list(CLASS_MODEL_FILES.keys()),
        help="Classe a detecter parmi : " + ", ".join(CLASS_MODEL_FILES.keys())
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    MODELS_DIR = Path(args.models_dir)

    if not Path(args.image).exists():
        raise FileNotFoundError(args.image)

    print("=" * 50)
    print("CLASSIFIEUR :", args.class_name)
    print("=" * 50)

    model = load_model(args.class_name)

    prob_map, mask, pct = predict_image(model, args.image, args.threshold)

    visualize(args.image, prob_map, mask, pct, args.class_name)

    print()
    print("=" * 50)
    print("TERMINE")


if __name__ == "__main__":
    main()
