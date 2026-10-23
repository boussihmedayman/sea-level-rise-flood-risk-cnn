# -*- coding: utf-8 -*-
"""
ANALYZE FLOOD RISK + VALEUR FONCIÈRE — MULTI-SCÉNARIOS → EXCEL
=================================================================
Adapté de analyze_flood_risk.py : au lieu d'analyser un seul dossier de
tuiles d'inondation, parcourt automatiquement tous les scénarios produits
par eec_flood_tiles_extractor.py (+0.2m, +0.3m, ..., +1.0m) et exporte les
résultats importants de chaque scénario dans un unique classeur Excel.

Pas de rapport HTML — sortie unique : un fichier .xlsx avec :
  - "Résumé global"      : 1 ligne par scénario (surface/valeur à risque)
  - "Détail par classe"  : 1 ligne par scénario x classe (Champs/Maisons/Forêt/Usines)
  - "Détail tuiles"      : 1 ligne par scénario x tuile (submersion globale)

Tous les chemins sont fixés directement dans le bloc CONFIG ci-dessous —
aucun argument n'est requis.

Usage :
  python src/analyze_flood_risk_multiscenario.py
"""

import re
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy import ndimage
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

# ─────────────────────────────────────────
# CONFIG — chemins (à ajuster si besoin)
# ─────────────────────────────────────────

CLASSES_DIR     = Path("results")                        # cartes de classes Tuile{id}_map.npy
FLOOD_BASE_DIR  = Path("outputs/eec_flood_tiles")         # sortie de eec_flood_tiles_extractor.py
PRICE_DIR       = Path("outputs/eec_price_tiles")         # tile_{id}_price.npy
OUTPUT_XLSX     = Path("flood_risk_multiscenario.xlsx")   # classeur Excel final
SCENARIOS       = None                                    # ex: ["+0.5m", "+1.0m"] ; None = tous

PIXEL_SIZE_M = 5

# price_map (tile_{id}_price.npy, produit par eec_land_pricev2.py) est en
# บาท/ตารางวา (Baht par "square wah"), l'unité native des prix Trésor —
# PAS en Baht/m². 1 ตารางวา (wah²) = 4 m² exactement (1 wah = 2 m).
# Il faut donc convertir la surface d'un pixel (m²) en wah² avant de la
# multiplier par le prix, sous peine de surestimer toutes les valeurs
# monétaires d'un facteur 4.
WAH2_IN_M2 = 4.0

CLASS_NAMES = {
    0: "Inconnu",
    1: "Riviere",
    2: "Mer",
    3: "Champs",
    4: "Maisons",
    5: "Foret",
    6: "Usines",
}

CLASS_BUFFER_PX = {
    4: 6,    # Maisons  : 30m
    6: 12,   # Usines   : 60m
    5: 0,
    3: 0,
}

CLASS_PRICE_USAGE = {
    3: "agricultural",   # Champs
    4: "residential",    # Maisons
    5: "agricultural",   # Foret
    6: "industrial",     # Usines
}

CLASSES_TO_ANALYZE = [3, 4, 5, 6]

ENCLAVE_MAX_PX = int(10 * 1_000_000 / PIXEL_SIZE_M ** 2)  # 400 000 px = 10 km²

SCENARIO_DIR_RE = re.compile(r"^\+\d+\.\d+m$")

# ─────────────────────────────────────────
# CONFIG — grille des tuiles (pour fusion inter-tuiles des zones à risque)
# ─────────────────────────────────────────

GRID_NCOLS = 16  # nombre de tuiles par ligne (tuile 0-15 = ligne 0, 16-31 = ligne 1, etc.)

WATER_CLASSES = {1, 2}  # Riviere, Mer : classes utilisées pour la distance à l'eau


# ─────────────────────────────────────────
# CONFIG — barèmes de coût des solutions de protection
# ─────────────────────────────────────────
# Sources (converties en Baht au taux ~33.3 ฿/$ US, juillet 2026) :
#  - Digue / remblai de protection : U.S. Army Corps of Engineers,
#    "Summary of Costs Associated with Levee-related Activities" (nov. 2023) —
#    coût de base de l'ouvrage en terre, ramené au m³ de matériau mis en place,
#    tout compris (terrassement, ingénierie, contingence).
#  - Remblai simple (rehaussement de terrain) : benchmarks internationaux de
#    terrassement 2025-2026 (remblai importé + compactage).
#  - Mangrove (restauration) : littérature internationale de restauration de
#    mangrove — moyenne mondiale ~9 739 $/ha, médiane ~2 099 $/ha (Goto et al.,
#    One Earth 2025, 249 projets dans 25 pays) ; pour l'Asie du Sud-Est
#    spécifiquement (delta indonésien), ~3 900 $/ha (World Bank, "Economics of
#    Large-scale Mangrove Conservation and Restoration in Indonesia"). On
#    retient ~5 000 $/ha comme ordre de grandeur régional SE Asie.
#    Largeur de ceinture efficace : la littérature technique (Deltares /
#    Communications Earth & Environment 2025) indique qu'une largeur minimale
#    d'environ 100 m est requise pour limiter réellement les forces
#    hydrauliques, et qu'une largeur de 500 m ou plus dissipe ~75% de
#    l'énergie des vagues entrantes.
#  - Relocalisation : pas de barème universel publié ; estimée forfaitairement
#    comme un multiple de la valeur foncière à risque (indemnisation +
#    démolition + frais de déménagement).
# Ces valeurs sont des ORDRES DE GRANDEUR à ajuster selon le contexte local
# (accès au site, disponibilité des matériaux, prix du foncier, etc.).

USD_TO_THB = 33.3

COST_DIGUE_THB_PER_M3   = 1700   # ฿/m³ de remblai de digue mis en place
DIGUE_CREST_WIDTH_M     = 3.0    # largeur de crête (≈10 ft, standard USACE)
DIGUE_SIDE_SLOPE        = 3.0    # pente des talus 1V:3H (run = 3 × hauteur)
DIGUE_FREEBOARD_M       = 0.5    # revanche de sécurité au-dessus du niveau d'eau du scénario

COST_REMBLAI_THB_PER_M3 = 1000   # ฿/m³ de remblai simple (rehaussement compacté)

COST_MANGROVE_THB_PER_HA = 166_500  # ≈ 5 000 $/ha (SE Asie), restauration de mangrove

RELOCALISATION_VALUE_MULT = 1.3  # coût de relocalisation = 130 % de la valeur foncière à risque

# ── Règles de sélection ──
COUT_MAX_PCT_VALEUR = 0.5   # si la meilleure solution technique coûte plus de 50% de la valeur à
                             # risque, on recommande la relocalisation plutôt que la protection

# ── Mangrove : éligibilité et dimensionnement ──
MANGROVE_MIN_WIDTH_M    = 100   # largeur minimale disponible entre la zone et la côte pour
                                 # qu'une mangrove soit hydrauliquement efficace ; en dessous, on
                                 # considère qu'il n'y a "pas assez de place" -> digue/remblai
MANGROVE_TARGET_WIDTH_M = 500   # largeur cible pour une bonne dissipation des vagues (~75%) ;
                                 # la ceinture plantée = min(place disponible, cette cible)

# ── Rivière assimilable à une côte (delta/estuaire large) ──
# Une rivière étroite ne permet pas d'implanter une mangrove utile ; une très
# grosse rivière (delta, estuaire large) se comporte hydrologiquement comme
# un littoral et peut en accueillir une. On estime la largeur locale de
# chaque tronçon de rivière (transformée de distance interne à la composante)
# et on ne le traite comme "côte" que si sa largeur estimée dépasse ce seuil.
RIVIERE_LARGE_MIN_WIDTH_M = 200

# NB : l'indice de compacité (périmètre / 2*sqrt(pi*surface), = 1 pour un cercle
# parfait, > 1 sinon) n'est pas utilisé comme filtre d'éligibilité : le
# compromis "forme allongée -> digue chère" est déjà capturé naturellement
# par la comparaison de coût digue (proportionnel au périmètre) vs remblai
# (proportionnel à la surface). Il est seulement reporté à titre indicatif
# dans les résultats.


# ─────────────────────────────────────────
# DÉCOUVERTE DES SCÉNARIOS
# ─────────────────────────────────────────

def discover_scenarios(flood_base_dir: Path, wanted: list | None = None) -> list:
    """Liste les sous-dossiers de scénarios (+0.2m, +0.3m, ...) contenant un index valide."""
    scenarios = []
    for d in sorted(flood_base_dir.iterdir()):
        if not d.is_dir() or not SCENARIO_DIR_RE.match(d.name):
            continue
        if not (d / "tiles_flood_index.json").exists():
            continue
        if wanted and d.name not in wanted:
            continue
        scenarios.append(d)

    if not scenarios:
        raise FileNotFoundError(
            f"Aucun scénario trouvé dans {flood_base_dir} "
            "(dossiers attendus : +0.2m, +0.3m, ..., avec tiles_flood_index.json)"
        )

    # tri numérique par seuil (+0.2m avant +1.0m)
    scenarios.sort(key=lambda d: float(d.name.strip("+m")))
    return scenarios


# ─────────────────────────────────────────
# CORRESPONDANCE TUILES (identique)
# ─────────────────────────────────────────

def find_pairs(classes_dir, flood_dir, price_dir):
    classes_dir = Path(classes_dir)
    flood_dir   = Path(flood_dir)
    price_dir   = Path(price_dir)

    index_path = flood_dir / "tiles_flood_index.json"
    with open(index_path) as f:
        index = json.load(f)

    pairs = []
    for entry in index:
        tile_id    = entry["id"]
        flood_path = flood_dir / entry["npy_file"]
        class_path = classes_dir / f"Tuile{tile_id}_map.npy"

        if not class_path.exists() or not flood_path.exists():
            continue

        price_path = price_dir / f"tile_{tile_id:03d}_price.npy"
        has_price  = price_path.exists()

        pairs.append({
            "tile_id"    : tile_id,
            "class_path" : class_path,
            "flood_path" : flood_path,
            "price_path" : price_path if has_price else None,
            "has_price"  : has_price,
            "flooded_pct": entry["flooded_pct"],
            "lat_min"    : entry["lat_min"],
            "lat_max"    : entry["lat_max"],
            "lon_min"    : entry["lon_min"],
            "lon_max"    : entry["lon_max"],
        })

    return pairs


# ─────────────────────────────────────────
# DÉTECTION DES ÎLOTS ENCLAVÉS (identique)
# ─────────────────────────────────────────

def compute_enclave_mask(flood_effective):
    not_flooded = ~flood_effective.astype(bool)
    labeled, n_components = ndimage.label(not_flooded)

    enclave_mask  = np.zeros(flood_effective.shape, dtype=bool)
    border_labels = set()

    border_labels.update(np.unique(labeled[0, :]))
    border_labels.update(np.unique(labeled[-1, :]))
    border_labels.update(np.unique(labeled[:, 0]))
    border_labels.update(np.unique(labeled[:, -1]))
    border_labels.discard(0)

    for lbl in range(1, n_components + 1):
        if lbl in border_labels:
            continue
        component = (labeled == lbl)
        if component.sum() < ENCLAVE_MAX_PX:
            enclave_mask |= component

    return enclave_mask


# ─────────────────────────────────────────
# ZONES À RISQUE (composantes connexes) + FUSION INTER-TUILES
# ─────────────────────────────────────────
# Contrairement à l'analyse par classe (ci-dessous), qui agrège toute une
# classe sur toute une tuile, ce qui suit isole chaque zone à risque
# individuelle (composante connexe), y compris lorsqu'elle traverse la
# frontière entre deux tuiles adjacentes de la grille, pour lui associer
# une géométrie propre (surface, périmètre, distance à l'eau) et une
# solution de protection recommandée.

def _local_perimeter(mask):
    """Nombre d'arêtes de pixels au contact de l'extérieur du masque (hors
    masque ou hors tuile). Unité : nombre de côtés de pixel — à multiplier
    par PIXEL_SIZE_M pour un périmètre en mètres. Les arêtes situées sur le
    bord de la tuile sont traitées ici comme "extérieures" ; celles qui
    correspondent en réalité à une continuité avec la tuile voisine sont
    corrigées ensuite dans merge_zones_across_tiles."""
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    up    = padded[:-2, 1:-1]
    down  = padded[2:, 1:-1]
    left  = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    edges = (mask & ~up).astype(np.int32) + (mask & ~down).astype(np.int32) \
          + (mask & ~left).astype(np.int32) + (mask & ~right).astype(np.int32)
    return int(edges.sum())


def compute_coast_mask(class_map):
    """Masque des pixels considérés comme "côte" pour l'éligibilité mangrove :
    toute la Mer (classe 2), plus les tronçons de Rivière (classe 1) dont la
    largeur locale estimée dépasse RIVIERE_LARGE_MIN_WIDTH_M (delta/estuaire
    large, hydrologiquement assimilable à un littoral). La largeur d'un
    tronçon est approximée par 2x la distance maximale, à l'intérieur de ce
    tronçon, à son propre bord (= largeur au niveau le plus large du chenal)."""
    mer_mask = (class_map == 2)

    riviere_mask = (class_map == 1)
    large_riviere_mask = np.zeros_like(riviere_mask)
    if riviere_mask.any():
        labeled, n_comp = ndimage.label(riviere_mask, structure=np.ones((3, 3), dtype=np.int32))
        min_width_px = RIVIERE_LARGE_MIN_WIDTH_M / PIXEL_SIZE_M
        for lbl in range(1, n_comp + 1):
            comp = (labeled == lbl)
            dist_in = ndimage.distance_transform_edt(comp)
            width_px_est = 2.0 * float(dist_in.max())
            if width_px_est >= min_width_px:
                large_riviere_mask |= comp

    return mer_mask | large_riviere_mask


def analyze_tile_zones(pair):
    """Identifie, pour une tuile, les zones à risque individuelles (composantes
    connexes du masque à risque, chaque classe étant mise à risque par SON
    PROPRE buffer, comme dans analyze_tile) et calcule pour chacune : surface,
    périmètre local, valeur foncière, distance à l'eau, et les labels de bord
    nécessaires à la fusion inter-tuiles."""
    class_map = np.load(pair["class_path"]).astype(np.int32)
    flood_map = np.load(pair["flood_path"]).astype(np.uint8)

    if flood_map.shape != class_map.shape:
        from PIL import Image
        flood_img = Image.fromarray(flood_map * 255, mode="L")
        flood_img = flood_img.resize((class_map.shape[1], class_map.shape[0]), Image.NEAREST)
        flood_map = (np.array(flood_img) > 128).astype(np.uint8)
    flood_bool = flood_map.astype(bool)

    price_map = None
    if pair["price_path"] is not None:
        price_map = np.load(pair["price_path"]).astype(np.float32)
        if price_map.shape != class_map.shape:
            from PIL import Image
            pm_img = Image.fromarray(price_map)
            price_map = np.array(pm_img.resize(
                (class_map.shape[1], class_map.shape[0]), Image.BILINEAR
            )).astype(np.float32)

    # union des zones dilatées par classe : sert UNIQUEMENT à détecter les
    # îlots enclavés (notion géographique, indépendante de la classe qui s'y
    # trouve -> on garde l'union ici, c'est correct).
    flood_union = flood_bool.copy()
    flood_dilated_per_cls = {}
    for cls_idx in CLASSES_TO_ANALYZE:
        buffer_px = CLASS_BUFFER_PX.get(cls_idx, 0)
        dilated = ndimage.binary_dilation(flood_bool, iterations=buffer_px) if buffer_px > 0 else flood_bool
        flood_dilated_per_cls[cls_idx] = dilated
        flood_union |= dilated
    enclave_mask = compute_enclave_mask(flood_union)

    # masque à risque : chaque classe n'est mise à risque que par SON PROPRE
    # buffer (identique à analyze_tile), pas par l'union de tous les buffers
    # -> évite qu'un Champ/Forêt (buffer 0m) proche d'une Usine (buffer 60m)
    # soit compté à tort comme à risque via le buffer de l'usine.
    risk_mask = np.zeros_like(flood_bool)
    for cls_idx in CLASSES_TO_ANALYZE:
        class_mask = (class_map == cls_idx)
        risk_mask |= class_mask & flood_dilated_per_cls[cls_idx]
    risk_mask |= np.isin(class_map, CLASSES_TO_ANALYZE) & enclave_mask

    # distance à l'eau (Mer + Rivière), calculée dans la tuile (approximation :
    # une zone proche d'un bord de tuile peut avoir de l'eau plus proche
    # encore dans la tuile voisine, non prise en compte ici)
    water_mask = np.isin(class_map, list(WATER_CLASSES))
    if water_mask.any():
        dist_to_water_px = ndimage.distance_transform_edt(~water_mask)
    else:
        dist_to_water_px = np.full(class_map.shape, np.inf, dtype=np.float32)

    # distance à la "côte" (mer + grosse rivière/delta) : sert uniquement à
    # juger de l'éligibilité de la solution mangrove
    coast_mask = compute_coast_mask(class_map)
    if coast_mask.any():
        dist_to_coast_px = ndimage.distance_transform_edt(~coast_mask)
    else:
        dist_to_coast_px = np.full(class_map.shape, np.inf, dtype=np.float32)

    labeled, n_zones = ndimage.label(risk_mask, structure=np.ones((3, 3), dtype=np.int32))

    pixel_m2   = PIXEL_SIZE_M ** 2
    pixel_wah2 = pixel_m2 / WAH2_IN_M2

    zones = {}
    for lbl in range(1, n_zones + 1):
        zmask = (labeled == lbl)
        area_px = int(zmask.sum())
        if area_px == 0:
            continue

        perim_px = _local_perimeter(zmask)
        value_thb = float((price_map[zmask] * pixel_wah2).sum()) if price_map is not None else 0.0

        dist_vals = dist_to_water_px[zmask]
        finite = dist_vals[np.isfinite(dist_vals)]
        min_dist_px = float(finite.min()) if finite.size > 0 else float("inf")

        coast_vals = dist_to_coast_px[zmask]
        coast_finite = coast_vals[np.isfinite(coast_vals)]
        min_dist_coast_px = float(coast_finite.min()) if coast_finite.size > 0 else float("inf")

        classes_present = {
            CLASS_NAMES[c]: int((class_map[zmask] == c).sum())
            for c in CLASSES_TO_ANALYZE if (class_map[zmask] == c).any()
        }

        zones[lbl] = {
            "area_px": area_px,
            "perim_px": perim_px,
            "value_thb": value_thb,
            "min_dist_water_px": min_dist_px,
            "min_dist_coast_px": min_dist_coast_px,
            "classes_present": classes_present,
        }

    return {
        "tile_id": pair["tile_id"],
        "shape": class_map.shape,
        "labeled": labeled,   # tableau 2D des labels locaux (0 = hors zone)
        "zones": zones,
    }


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def merge_zones_across_tiles(tile_zone_results, ncols=GRID_NCOLS):
    """Fusionne les zones à risque qui traversent la frontière entre deux
    tuiles adjacentes de la grille (voisin de droite = même ligne, voisin du
    bas = ligne suivante), et corrige le périmètre en conséquence (les arêtes
    partagées à la frontière ne sont plus des bords extérieurs de la zone
    fusionnée)."""
    by_id = {t["tile_id"]: t for t in tile_zone_results}
    uf = _UnionFind()
    edge_correction = {}  # paire de clés (tuile, label) triée -> nb d'arêtes partagées

    def key(tile_id, lbl):
        return (tile_id, lbl)

    for t in tile_zone_results:
        tid = t["tile_id"]
        labeled = t["labeled"]
        row, col = divmod(tid, ncols)

        # voisin de droite (même ligne de la grille)
        right_id = tid + 1
        if col + 1 < ncols and right_id in by_id:
            other = by_id[right_id]
            h = min(labeled.shape[0], other["labeled"].shape[0])
            col_a = labeled[:h, -1]
            col_b = other["labeled"][:h, 0]
            match_idx = np.where((col_a > 0) & (col_b > 0))[0]
            for i in match_idx:
                la, lb = int(col_a[i]), int(col_b[i])
                ka, kb = key(tid, la), key(right_id, lb)
                uf.union(ka, kb)
                pair = tuple(sorted((ka, kb)))
                edge_correction[pair] = edge_correction.get(pair, 0) + 1

        # voisin du bas (ligne suivante de la grille)
        below_id = tid + ncols
        if below_id in by_id:
            other = by_id[below_id]
            w = min(labeled.shape[1], other["labeled"].shape[1])
            row_a = labeled[-1, :w]
            row_b = other["labeled"][0, :w]
            match_idx = np.where((row_a > 0) & (row_b > 0))[0]
            for j in match_idx:
                la, lb = int(row_a[j]), int(row_b[j])
                ka, kb = key(tid, la), key(below_id, lb)
                uf.union(ka, kb)
                pair = tuple(sorted((ka, kb)))
                edge_correction[pair] = edge_correction.get(pair, 0) + 1

    merged = {}
    for t in tile_zone_results:
        tid = t["tile_id"]
        for lbl, stats in t["zones"].items():
            root = uf.find(key(tid, lbl))
            m = merged.setdefault(root, {
                "area_px": 0, "perim_px": 0, "value_thb": 0.0,
                "min_dist_water_px": float("inf"), "min_dist_coast_px": float("inf"),
                "tiles": set(),
                "classes_present": {},
            })
            m["area_px"]   += stats["area_px"]
            m["perim_px"]  += stats["perim_px"]
            m["value_thb"] += stats["value_thb"]
            m["min_dist_water_px"] = min(m["min_dist_water_px"], stats["min_dist_water_px"])
            m["min_dist_coast_px"] = min(m["min_dist_coast_px"], stats["min_dist_coast_px"])
            m["tiles"].add(tid)
            for cname, cnt in stats["classes_present"].items():
                m["classes_present"][cname] = m["classes_present"].get(cname, 0) + cnt

    # correction du périmètre : chaque arête partagée entre deux tuiles a été
    # comptée deux fois comme "extérieure" (une fois de chaque côté) alors
    # qu'elle est désormais interne à la zone fusionnée -> on la retire deux fois.
    for (ka, kb), n_shared in edge_correction.items():
        root = uf.find(ka)
        if root in merged:
            merged[root]["perim_px"] -= 2 * n_shared

    return list(merged.values())


# ─────────────────────────────────────────
# SÉLECTION DE LA SOLUTION DE PROTECTION
# ─────────────────────────────────────────

def cost_digue_thb(perim_m, threshold_m):
    hauteur_m = threshold_m + DIGUE_FREEBOARD_M
    section_m2 = DIGUE_CREST_WIDTH_M * hauteur_m + DIGUE_SIDE_SLOPE * hauteur_m ** 2
    volume_m3 = section_m2 * perim_m
    return volume_m3 * COST_DIGUE_THB_PER_M3


def cost_remblai_thb(area_m2, threshold_m):
    hauteur_m = threshold_m + DIGUE_FREEBOARD_M
    volume_m3 = area_m2 * hauteur_m
    return volume_m3 * COST_REMBLAI_THB_PER_M3


def cost_mangrove_thb(area_m2):
    area_ha = area_m2 / 10_000
    return area_ha * COST_MANGROVE_THB_PER_HA


def cost_relocalisation_thb(value_thb):
    return value_thb * RELOCALISATION_VALUE_MULT


def dominant_class(classes_present: dict) -> str | None:
    """Classe majoritaire (en pixels) présente dans une zone à risque."""
    if not classes_present:
        return None
    return max(classes_present, key=classes_present.get)


def select_solution(area_m2, perim_m, dist_coast_m, value_thb, threshold_m, classes_present):
    """Arbre de décision pour la solution de protection recommandée :

    1. S'il y a assez de place entre la zone et la côte (mer, ou grosse
       rivière/delta assimilable à une côte) pour une mangrove
       hydrauliquement efficace (>= MANGROVE_MIN_WIDTH_M) -> Mangrove.
    2. Sinon (pas assez de place, ou pas de côte à proximité du tout) :
       - si la zone est majoritairement des Champs -> Remblai (rehaussement
         de terrain, pas de digue sur des champs) ;
       - sinon -> Digue.
    3. Dans tous les cas, si le coût de la solution retenue dépasse
       COUT_MAX_PCT_VALEUR de la valeur foncière à risque -> Relocalisation.
    """
    compacite = perim_m / (2 * math.sqrt(math.pi * area_m2)) if area_m2 > 0 else float("inf")

    mangrove_eligible = math.isfinite(dist_coast_m) and dist_coast_m >= MANGROVE_MIN_WIDTH_M

    if mangrove_eligible:
        largeur_m = min(dist_coast_m, MANGROVE_TARGET_WIDTH_M)
        area_mangrove_m2 = perim_m * largeur_m
        candidates = {"Mangrove": cost_mangrove_thb(area_mangrove_m2)}
        best_name = "Mangrove"
    else:
        cls = dominant_class(classes_present)
        if cls == "Champs":
            candidates = {"Remblai": cost_remblai_thb(area_m2, threshold_m)}
            best_name = "Remblai"
        else:
            candidates = {"Digue": cost_digue_thb(perim_m, threshold_m)}
            best_name = "Digue"

    best_cost = candidates[best_name]

    if value_thb > 0 and best_cost > COUT_MAX_PCT_VALEUR * value_thb:
        return "Relocalisation", cost_relocalisation_thb(value_thb), candidates, compacite

    return best_name, best_cost, candidates, compacite


def analyze_scenario_zones(scenario_dir, classes_dir, price_dir, n_workers, threshold_m, ncols=GRID_NCOLS):
    """Identifie les zones à risque fusionnées inter-tuiles d'un scénario et
    leur associe une solution de protection recommandée avec son coût estimé."""
    pairs = find_pairs(classes_dir, scenario_dir, price_dir)
    if not pairs:
        return []

    with mp.Pool(processes=n_workers) as pool:
        tile_zone_results = pool.map(analyze_tile_zones, pairs)

    merged_zones = merge_zones_across_tiles(tile_zone_results, ncols)

    px_m2 = PIXEL_SIZE_M ** 2
    zone_reports = []
    for i, z in enumerate(sorted(merged_zones, key=lambda m: -m["value_thb"]), start=1):
        area_m2 = z["area_px"] * px_m2
        perim_m = z["perim_px"] * PIXEL_SIZE_M
        dist_m  = z["min_dist_water_px"] * PIXEL_SIZE_M if math.isfinite(z["min_dist_water_px"]) else None
        dist_coast_m = z["min_dist_coast_px"] * PIXEL_SIZE_M if math.isfinite(z["min_dist_coast_px"]) else None

        solution, cost_thb, candidates, compacite = select_solution(
            area_m2, perim_m,
            dist_coast_m if dist_coast_m is not None else float("inf"),
            z["value_thb"], threshold_m, z["classes_present"],
        )

        zone_reports.append({
            "zone_id"        : i,
            "n_tiles"        : len(z["tiles"]),
            "tiles"          : sorted(z["tiles"]),
            "area_km2"       : round(area_m2 / 1_000_000, 4),
            "perim_m"        : round(perim_m, 1),
            "compacite"      : round(compacite, 3) if math.isfinite(compacite) else None,
            "dist_water_m"   : round(dist_m, 1) if dist_m is not None else None,
            "dist_coast_m"   : round(dist_coast_m, 1) if dist_coast_m is not None else None,
            "classe_dominante": dominant_class(z["classes_present"]),
            "value_mdthb"     : round(z["value_thb"] / 1_000_000_000, 6),
            "solution"        : solution,
            "cost_mdthb"      : round(cost_thb / 1_000_000_000, 6),
            "candidates_mdthb": {k: round(v / 1_000_000_000, 6) for k, v in candidates.items()},
            "classes_present" : z["classes_present"],
        })

    return zone_reports


# ─────────────────────────────────────────
# ANALYSE D'UNE TUILE (identique)
# ─────────────────────────────────────────

def analyze_tile(pair):
    class_map = np.load(pair["class_path"]).astype(np.int32)
    flood_map = np.load(pair["flood_path"]).astype(np.uint8)

    if flood_map.shape != class_map.shape:
        from PIL import Image
        flood_img = Image.fromarray(flood_map * 255, mode="L")
        flood_img = flood_img.resize((class_map.shape[1], class_map.shape[0]), Image.NEAREST)
        flood_map = (np.array(flood_img) > 128).astype(np.uint8)

    flood_bool = flood_map.astype(bool)

    price_map = None
    if pair["price_path"] is not None:
        price_map = np.load(pair["price_path"]).astype(np.float32)
        if price_map.shape != class_map.shape:
            from PIL import Image
            pm_img = Image.fromarray(price_map)
            price_map = np.array(pm_img.resize(
                (class_map.shape[1], class_map.shape[0]), Image.BILINEAR
            )).astype(np.float32)

    flood_union = flood_bool.copy()
    flood_dilated_per_cls = {}

    for cls_idx in CLASSES_TO_ANALYZE:
        buffer_px = CLASS_BUFFER_PX.get(cls_idx, 0)
        if buffer_px > 0:
            dilated = ndimage.binary_dilation(flood_bool, iterations=buffer_px)
        else:
            dilated = flood_bool
        flood_dilated_per_cls[cls_idx] = dilated
        flood_union |= dilated

    enclave_mask = compute_enclave_mask(flood_union)

    results = {}
    pixel_m2  = PIXEL_SIZE_M ** 2
    pixel_wah2 = pixel_m2 / WAH2_IN_M2   # surface d'un pixel en ตารางวา, pour multiplier price_map (บาท/ตร.ว.)

    for cls_idx in CLASSES_TO_ANALYZE:
        class_mask = (class_map == cls_idx)
        total_px   = int(class_mask.sum())
        buffer_px  = CLASS_BUFFER_PX.get(cls_idx, 0)
        usage      = CLASS_PRICE_USAGE.get(cls_idx, "agricultural")

        if total_px == 0:
            results[cls_idx] = {
                "total_px": 0, "total_m2": 0,
                "flooded_px": 0, "flooded_pct": 0.0,
                "buffer_px_area": 0, "buffer_pct": 0.0,
                "enclave_px": 0, "enclave_pct": 0.0,
                "at_risk_px": 0, "at_risk_pct": 0.0,
                "buffer_px": buffer_px, "buffer_m": buffer_px * PIXEL_SIZE_M,
                "total_value_thb": 0.0,
                "flooded_value_thb": 0.0, "buffer_value_thb": 0.0, "enclave_value_thb": 0.0,
                "risk_value_thb": 0.0, "price_usage": usage,
            }
            continue

        flood_dilated = flood_dilated_per_cls[cls_idx]

        # 3 catégories mutuellement exclusives (leur somme = zone à risque totale) :
        direct_mask  = class_mask & flood_bool                        # inondé directement
        buffer_mask  = class_mask & flood_dilated & ~flood_bool       # zone tampon (buffer only)
        encl_mask    = class_mask & enclave_mask                      # îlot enclavé (déjà disjoint des 2 autres)
        at_risk_mask = direct_mask | buffer_mask | encl_mask

        flooded_px     = int(direct_mask.sum())
        buffer_area_px = int(buffer_mask.sum())
        enclave_px     = int(encl_mask.sum())
        at_risk_px     = int(at_risk_mask.sum())

        if price_map is not None:
            total_value   = float((price_map[class_mask]  * pixel_wah2).sum())
            flooded_value = float((price_map[direct_mask] * pixel_wah2).sum())
            buffer_value  = float((price_map[buffer_mask] * pixel_wah2).sum())
            enclave_value = float((price_map[encl_mask]   * pixel_wah2).sum())
            risk_value    = flooded_value + buffer_value + enclave_value
        else:
            total_value = flooded_value = buffer_value = enclave_value = risk_value = 0.0

        results[cls_idx] = {
            "total_px"         : total_px,
            "total_m2"         : total_px * pixel_m2,
            "flooded_px"       : flooded_px,
            "flooded_pct"      : round(100 * flooded_px / total_px, 2),
            "buffer_px_area"   : buffer_area_px,
            "buffer_pct"       : round(100 * buffer_area_px / total_px, 2),
            "enclave_px"       : enclave_px,
            "enclave_pct"      : round(100 * enclave_px / total_px, 2),
            "at_risk_px"       : at_risk_px,
            "at_risk_pct"      : round(100 * at_risk_px / total_px, 2),
            "buffer_px"        : buffer_px,
            "buffer_m"         : buffer_px * PIXEL_SIZE_M,
            "total_value_thb"  : round(total_value, 0),
            "flooded_value_thb": round(flooded_value, 0),
            "buffer_value_thb" : round(buffer_value, 0),
            "enclave_value_thb": round(enclave_value, 0),
            "risk_value_thb"   : round(risk_value, 0),
            "price_usage"      : usage,
        }

    return results


# ─────────────────────────────────────────
# AGRÉGATION (identique)
# ─────────────────────────────────────────

def aggregate_results(all_results):
    totals = {cls_idx: {
        "total_px": 0, "flooded_px": 0, "buffer_px_area": 0, "enclave_px": 0, "at_risk_px": 0,
        "total_value_thb": 0.0, "flooded_value_thb": 0.0, "buffer_value_thb": 0.0,
        "enclave_value_thb": 0.0, "risk_value_thb": 0.0,
        "buffer_m": CLASS_BUFFER_PX.get(cls_idx, 0) * PIXEL_SIZE_M,
        "price_usage": CLASS_PRICE_USAGE.get(cls_idx, "agricultural"),
    } for cls_idx in CLASSES_TO_ANALYZE}

    for tile_result in all_results:
        for cls_idx, stats in tile_result.items():
            totals[cls_idx]["total_px"]          += stats["total_px"]
            totals[cls_idx]["flooded_px"]        += stats["flooded_px"]
            totals[cls_idx]["buffer_px_area"]    += stats["buffer_px_area"]
            totals[cls_idx]["enclave_px"]        += stats["enclave_px"]
            totals[cls_idx]["at_risk_px"]        += stats["at_risk_px"]
            totals[cls_idx]["total_value_thb"]   += stats["total_value_thb"]
            totals[cls_idx]["flooded_value_thb"] += stats["flooded_value_thb"]
            totals[cls_idx]["buffer_value_thb"]  += stats["buffer_value_thb"]
            totals[cls_idx]["enclave_value_thb"] += stats["enclave_value_thb"]
            totals[cls_idx]["risk_value_thb"]    += stats["risk_value_thb"]

    for cls_idx in CLASSES_TO_ANALYZE:
        t     = totals[cls_idx]
        total = t["total_px"]
        px_m2 = PIXEL_SIZE_M ** 2
        if total > 0:
            t["flooded_pct"]        = round(100 * t["flooded_px"]     / total, 2)
            t["buffer_pct"]         = round(100 * t["buffer_px_area"] / total, 2)
            t["enclave_pct"]        = round(100 * t["enclave_px"]     / total, 2)
            t["at_risk_pct"]        = round(100 * t["at_risk_px"]     / total, 2)
            t["total_km2"]          = round(total                  * px_m2 / 1_000_000, 2)
            t["flooded_km2"]        = round(t["flooded_px"]       * px_m2 / 1_000_000, 2)
            t["buffer_km2"]         = round(t["buffer_px_area"]   * px_m2 / 1_000_000, 2)
            t["enclave_km2"]        = round(t["enclave_px"]       * px_m2 / 1_000_000, 2)
            t["at_risk_km2"]        = round(t["at_risk_px"]       * px_m2 / 1_000_000, 2)
            t["total_value_mdthb"]   = round(t["total_value_thb"]   / 1_000_000_000, 5)
            t["flooded_value_mdthb"] = round(t["flooded_value_thb"] / 1_000_000_000, 5)
            t["buffer_value_mdthb"]  = round(t["buffer_value_thb"]  / 1_000_000_000, 5)
            t["enclave_value_mdthb"] = round(t["enclave_value_thb"] / 1_000_000_000, 5)
            t["risk_value_mdthb"]    = round(t["risk_value_thb"]    / 1_000_000_000, 5)
        else:
            for k in ["flooded_pct", "buffer_pct", "enclave_pct", "at_risk_pct",
                      "total_km2", "flooded_km2", "buffer_km2", "enclave_km2", "at_risk_km2",
                      "total_value_mdthb", "flooded_value_mdthb", "buffer_value_mdthb",
                      "enclave_value_mdthb", "risk_value_mdthb"]:
                t[k] = 0.0

    return totals


# ─────────────────────────────────────────
# ANALYSE D'UN SCÉNARIO COMPLET
# ─────────────────────────────────────────

def analyze_scenario(scenario_dir: Path, classes_dir, price_dir, n_workers: int) -> dict:
    label = scenario_dir.name
    pairs = find_pairs(classes_dir, scenario_dir, price_dir)

    if not pairs:
        print(f"  [{label}] Aucune paire tuile trouvée, scénario ignoré.")
        return None

    n_with_price = sum(1 for p in pairs if p["has_price"])
    print(f"  [{label}] {len(pairs)} tuiles analysées ({n_with_price} avec prix fonciers)...")

    with mp.Pool(processes=n_workers) as pool:
        results_list = pool.map(analyze_tile, pairs)

    tile_reports = []
    for pair, result in zip(pairs, results_list):
        tile_reports.append({
            "tile_id"         : pair["tile_id"],
            "lat_min"         : pair["lat_min"],
            "lat_max"         : pair["lat_max"],
            "lon_min"         : pair["lon_min"],
            "lon_max"         : pair["lon_max"],
            "flood_pct_global": pair["flooded_pct"],
            "classes"         : {CLASS_NAMES[k]: v for k, v in result.items()},
        })

    totals = aggregate_results(results_list)

    threshold_m = float(label.strip("+m"))
    zone_reports = analyze_scenario_zones(scenario_dir, classes_dir, price_dir, n_workers, threshold_m)

    return {
        "label"       : label,
        "threshold_m" : threshold_m,
        "n_tiles"     : len(pairs),
        "totals"      : totals,
        "tile_reports": tile_reports,
        "zone_reports": zone_reports,
    }


# ─────────────────────────────────────────
# EXPORT EXCEL
# ─────────────────────────────────────────

HEADER_FILL = PatternFill("solid", start_color="1F2937", end_color="1F2937")
HEADER_FONT = Font(name="Arial", bold=True, color="FFFFFF")
BASE_FONT   = Font(name="Arial")
TITLE_FONT  = Font(name="Arial", bold=True, size=14)


def _style_header(ws, row=1):
    for cell in ws[row]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _autofit(ws, min_width=10, max_width=42):
    for col_cells in ws.columns:
        length = max((len(str(c.value)) for c in col_cells if c.value is not None), default=0)
        col_letter = get_column_letter(col_cells[0].column)
        ws.column_dimensions[col_letter].width = min(max(length + 2, min_width), max_width)


def aggregate_zone_solutions(zone_reports: list) -> tuple:
    """Agrège les coûts des solutions retenues par zone à risque, à la fois
    globalement (pour la feuille 'Résumé global') et par classe dominante
    (pour la feuille 'Détail par classe').

    Retourne (global_agg, by_class) où chacun est un dict :
      - cost_total_mdthb : somme des coûts des solutions retenues (Md฿)
      - area_reloc_km2   : surface des zones dont la solution retenue est
                            "Relocalisation" (km²)
      - cost_reloc_mdthb : coût de relocalisation de ces zones (Md฿)
    """
    def _new(): return {"cost_total_mdthb": 0.0, "area_reloc_km2": 0.0, "cost_reloc_mdthb": 0.0}

    global_agg = _new()
    by_class = {}

    for z in zone_reports:
        cost = z["cost_mdthb"]
        global_agg["cost_total_mdthb"] += cost

        cls = z.get("classe_dominante") or "Inconnu"
        c = by_class.setdefault(cls, _new())
        c["cost_total_mdthb"] += cost

        if z["solution"] == "Relocalisation":
            global_agg["area_reloc_km2"]   += z["area_km2"]
            global_agg["cost_reloc_mdthb"] += cost
            c["area_reloc_km2"]   += z["area_km2"]
            c["cost_reloc_mdthb"] += cost

    return global_agg, by_class


def build_workbook(scenario_results: list, output_path: Path):
    wb = Workbook()

    # ── Feuille 1 : Résumé global (1 ligne par scénario) ──────────
    ws = wb.active
    ws.title = "Résumé global"
    headers = [
        "Scénario", "Seuil (m)", "Tuiles analysées",
        "Surface totale (km²)", "Surface inondée (km²)",
        "Surface à risque (km²)",
        "Valeur foncière totale (Md฿)", "Valeur à risque (Md฿)",
        "Coût total solutions (Md฿)",
        "Surface relocalisée (km²)", "Coût relocalisation (Md฿)",
        "Net économisé (Md฿)",
    ]
    ws.append(headers)
    _style_header(ws)

    for sc in scenario_results:
        totals = sc["totals"]
        total_km2   = sum(t["total_km2"] for t in totals.values())
        flooded_km2 = sum(t["flooded_km2"] for t in totals.values())
        risk_km2    = sum(t["at_risk_km2"] for t in totals.values())
        total_val   = sum(t["total_value_mdthb"] for t in totals.values())
        risk_val    = sum(t["risk_value_mdthb"] for t in totals.values())

        global_agg, _ = aggregate_zone_solutions(sc.get("zone_reports", []))
        cost_total  = global_agg["cost_total_mdthb"]
        area_reloc  = global_agg["area_reloc_km2"]
        cost_reloc  = global_agg["cost_reloc_mdthb"]
        net_saved   = round(risk_val - cost_total, 5)

        ws.append([
            sc["label"], sc["threshold_m"], sc["n_tiles"],
            round(total_km2, 2), round(flooded_km2, 2),
            round(risk_km2, 2),
            round(total_val, 5), round(risk_val, 5),
            round(cost_total, 5),
            round(area_reloc, 4), round(cost_reloc, 5),
            net_saved,
        ])

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = BASE_FONT
    _autofit(ws)
    ws.freeze_panes = "A2"

    # ── Feuille 2 : Détail par classe (1 ligne par scénario x classe) ──
    ws2 = wb.create_sheet("Détail par classe")
    headers2 = [
        "Scénario", "Seuil (m)", "Classe", "Usage", "Buffer (m)",
        "Surface totale (km²)",
        "Inondé direct (km²)",
        "Zone tampon (km²)",
        "Îlot enclavé (km²)",
        "Total à risque (km²)",
        "Valeur totale (Md฿)",
        "Valeur inondé direct (Md฿)", "Valeur zone tampon (Md฿)", "Valeur îlot enclavé (Md฿)",
        "Valeur totale à risque (Md฿)",
        "Coût total solutions (Md฿)",
        "Surface relocalisée (km²)", "Coût relocalisation (Md฿)",
        "Net économisé (Md฿)",
    ]
    ws2.append(headers2)
    _style_header(ws2)

    for sc in scenario_results:
        _, by_class = aggregate_zone_solutions(sc.get("zone_reports", []))
        for cls_idx in CLASSES_TO_ANALYZE:
            t = sc["totals"][cls_idx]
            if t["total_px"] == 0:
                continue
            cls_name  = CLASS_NAMES[cls_idx]
            c         = by_class.get(cls_name, {"cost_total_mdthb": 0.0, "area_reloc_km2": 0.0, "cost_reloc_mdthb": 0.0})
            net_saved = round(t["risk_value_mdthb"] - c["cost_total_mdthb"], 5)
            ws2.append([
                sc["label"], sc["threshold_m"], cls_name, t["price_usage"], t["buffer_m"],
                t["total_km2"],
                t["flooded_km2"],
                t["buffer_km2"],
                t["enclave_km2"],
                t["at_risk_km2"],
                t["total_value_mdthb"],
                t["flooded_value_mdthb"], t["buffer_value_mdthb"], t["enclave_value_mdthb"],
                t["risk_value_mdthb"],
                round(c["cost_total_mdthb"], 5),
                round(c["area_reloc_km2"], 4), round(c["cost_reloc_mdthb"], 5),
                net_saved,
            ])

    for row in ws2.iter_rows(min_row=2):
        for cell in row:
            cell.font = BASE_FONT
    _autofit(ws2)
    ws2.freeze_panes = "A2"

    # ── Feuille 3 : Détail tuiles (1 ligne par scénario x tuile) ──
    ws3 = wb.create_sheet("Détail tuiles")
    headers3 = [
        "Scénario", "Seuil (m)", "Tuile ID", "Lat min", "Lat max", "Lon min", "Lon max",
    ]
    ws3.append(headers3)
    _style_header(ws3)

    for sc in scenario_results:
        for t in sc["tile_reports"]:
            ws3.append([
                sc["label"], sc["threshold_m"], t["tile_id"],
                t["lat_min"], t["lat_max"], t["lon_min"], t["lon_max"],
            ])

    for row in ws3.iter_rows(min_row=2):
        for cell in row:
            cell.font = BASE_FONT
    _autofit(ws3)
    ws3.freeze_panes = "A2"

    # ── Feuille 4 : Zones & solutions (1 ligne par zone à risque fusionnée) ──
    ws4 = wb.create_sheet("Zones & solutions")
    headers4 = [
        "Scénario", "Seuil (m)", "Zone ID", "Nb tuiles", "Tuiles",
        "Surface (km²)", "Périmètre (m)", "Compacité (péri/aire)",
        "Distance à l'eau (m)", "Distance à la côte (m)", "Classe dominante",
        "Valeur à risque (Md฿)",
        "Solution recommandée", "Coût estimé (Md฿)",
        "Coût Mangrove (Md฿)", "Coût Digue (Md฿)", "Coût Remblai (Md฿)",
        "Classes présentes",
    ]
    ws4.append(headers4)
    _style_header(ws4)

    for sc in scenario_results:
        for z in sc.get("zone_reports", []):
            cand = z["candidates_mdthb"]
            classes_str = ", ".join(f"{k} ({v})" for k, v in z["classes_present"].items())
            ws4.append([
                sc["label"], sc["threshold_m"], z["zone_id"], z["n_tiles"],
                ", ".join(str(t) for t in z["tiles"]),
                z["area_km2"], z["perim_m"], z["compacite"],
                z["dist_water_m"], z["dist_coast_m"], z["classe_dominante"],
                z["value_mdthb"],
                z["solution"], z["cost_mdthb"],
                cand.get("Mangrove"), cand.get("Digue"), cand.get("Remblai"),
                classes_str,
            ])

    for row in ws4.iter_rows(min_row=2):
        for cell in row:
            cell.font = BASE_FONT
    _autofit(ws4)
    ws4.freeze_panes = "A2"

    wb.save(output_path)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("ANALYSE RISQUE INONDATION + VALEUR FONCIÈRE — MULTI-SCÉNARIOS")
    print("=" * 60)
    print(f"  Classes  : {CLASSES_DIR}")
    print(f"  Flood    : {FLOOD_BASE_DIR}")
    print(f"  Prix     : {PRICE_DIR}")
    print(f"  Sortie   : {OUTPUT_XLSX}\n")

    scenarios = discover_scenarios(FLOOD_BASE_DIR, wanted=SCENARIOS)
    print(f"Scénarios trouvés : {[d.name for d in scenarios]}\n")

    n_workers = min(mp.cpu_count(), 8)
    scenario_results = []

    for scenario_dir in scenarios:
        result = analyze_scenario(scenario_dir, CLASSES_DIR, PRICE_DIR, n_workers)
        if result is not None:
            scenario_results.append(result)

    if not scenario_results:
        print("Aucun scénario n'a pu être analysé. Vérifiez les chemins dans le bloc CONFIG.")
        return

    build_workbook(scenario_results, OUTPUT_XLSX)

    print("\n" + "=" * 60)
    print(f"Classeur Excel sauvegardé : {OUTPUT_XLSX.resolve()}")
    print(f"  - Résumé global      : {len(scenario_results)} scénarios")
    print(f"  - Détail par classe  : {sum(len(CLASSES_TO_ANALYZE) for _ in scenario_results)} lignes max")
    print(f"  - Détail tuiles      : {sum(len(s['tile_reports']) for s in scenario_results)} lignes")
    print(f"  - Zones & solutions  : {sum(len(s.get('zone_reports', [])) for s in scenario_results)} zones à risque")
    print("=" * 60)


if __name__ == "__main__":
    main()