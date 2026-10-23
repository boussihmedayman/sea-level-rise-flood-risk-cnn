"""
GISTDA Ecoplant - Script de téléchargement automatisé
4 cultures × ~79 fichiers = ~316 téléchargements
"""

import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

# ─── CONFIG ────────────────────────────────────────────────────────────────────
URL           = "https://ecoplant.gistda.or.th"
DOWNLOAD_DIR  = os.path.join(os.getcwd(), "ecoplant_data")
WAIT          = 15
PAUSE         = 1.5
DOWNLOAD_WAIT = 6

CULTURES = ["ข้าว", "ข้าวโพด", "มันสำปะหลัง", "อ้อย"]

# Items parasites à ignorer (conteneurs, pas des fichiers)
SKIP_TEXTS = {
    "ดาวน์โหลดข้อมูลพืช", "รายงานรายจังหวัด", "Shapefile",
    "แผนที่รายจังหวัด", "ชั้นข้อมูลพืชในประเทศไทย",
    "ข้าว", "ข้าวโพด", "มันสำปะหลัง", "อ้อย", "ยางพารา", "ปาล์มน้ำมัน",
}
# ───────────────────────────────────────────────────────────────────────────────


def make_driver(download_dir: str) -> webdriver.Chrome:
    os.makedirs(download_dir, exist_ok=True)
    options = Options()
    options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    })
    options.add_argument("--window-size=1400,900")
    options.add_argument("--lang=th")
    driver = webdriver.Chrome(options=options)
    return driver


def safe_click(driver, el):
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    time.sleep(0.3)
    driver.execute_script("arguments[0].click();", el)


def wait_overlay_gone(driver, timeout=10):
    try:
        WebDriverWait(driver, timeout).until(
            EC.invisibility_of_element_located((By.CSS_SELECTOR, "div.v-overlay__scrim"))
        )
    except Exception:
        pass
    time.sleep(0.5)


def close_modal(driver):
    print("  → Fermeture modal...")
    closed = False
    for xpath in [
        "//button[contains(@style,'239') or contains(@style,'220') or contains(@style,'red')]",
        "//*[@aria-label='close' or @aria-label='Close']",
        "//div[contains(@class,'v-overlay__content')]//button[contains(@class,'v-btn')][not(contains(.,'ดาวน์โหลด'))]",
    ]:
        try:
            for el in driver.find_elements(By.XPATH, xpath):
                if el.is_displayed():
                    safe_click(driver, el)
                    closed = True
                    break
            if closed:
                break
        except Exception:
            pass
    if not closed:
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
    wait_overlay_gone(driver)
    time.sleep(PAUSE)


# ─── ÉTAPE 1 ───────────────────────────────────────────────────────────────────
def open_download_menu(driver):
    print("  [1] Clic bouton ดาวน์โหลด navbar...")
    wait_overlay_gone(driver)
    el = None
    deadline = time.time() + WAIT
    while time.time() < deadline and el is None:
        for xpath in [
            "//button[contains(@class,'v-btn')][contains(., 'ดาวน์โหลด')]",
            "//*[contains(@class,'v-btn')][contains(., 'ดาวน์โหลด')]",
            "//*[contains(text(),'ดาวน์โหลด')]",
        ]:
            try:
                for c in driver.find_elements(By.XPATH, xpath):
                    if c.is_displayed():
                        el = c
                        break
            except Exception:
                pass
        if el is None:
            time.sleep(0.5)
    if el is None:
        raise Exception("Bouton ดาวน์โหลด introuvable")
    safe_click(driver, el)
    time.sleep(PAUSE)


# ─── ÉTAPE 2 ───────────────────────────────────────────────────────────────────
def select_culture(driver, culture_name: str):
    print(f"  [2] Sélection '{culture_name}'...")
    el = None
    deadline = time.time() + WAIT
    while time.time() < deadline and el is None:
        for xpath in [
            f"(//div[contains(@class,'btn-card-download')][.//span[normalize-space()='{culture_name}']])[1]",
            f"(//span[contains(@class,'font-btn-card-download')][normalize-space()='{culture_name}'])[1]/..",
        ]:
            try:
                for c in driver.find_elements(By.XPATH, xpath):
                    if c.is_displayed():
                        el = c
                        break
            except Exception:
                pass
        if el is None:
            time.sleep(0.5)
    if el is None:
        raise Exception(f"Culture '{culture_name}' introuvable")
    print(f"     text='{el.text[:40]}'")
    safe_click(driver, el)
    time.sleep(PAUSE)


# ─── ÉTAPE 3 ───────────────────────────────────────────────────────────────────
def open_xlsx_folder(driver):
    print("  [3] Clic dossier รายงานรายจังหวัด (.xlsx)...")
    el = None
    deadline = time.time() + WAIT
    while time.time() < deadline and el is None:
        for xpath in [
            "//div[contains(@class,'right-card-download')]//*[normalize-space()='รายงานรายจังหวัด']",
            "//*[normalize-space()='รายงานรายจังหวัด']",
            "//span[normalize-space()='รายงานรายจังหวัด']",
            "//div[normalize-space()='รายงานรายจังหวัด']",
            "//*[./span[normalize-space()='รายงานรายจังหวัด'] or ./div[normalize-space()='รายงานรายจังหวัด']]",
        ]:
            try:
                for c in driver.find_elements(By.XPATH, xpath):
                    txt = c.text.strip()
                    if c.is_displayed() and txt == 'รายงานรายจังหวัด':
                        el = c
                        break
                    if c.is_displayed() and 'รายงานรายจังหวัด' in txt and 'Shapefile' not in txt and '.pdf' not in txt:
                        el = c
            except Exception:
                pass
        if el is None:
            time.sleep(0.5)
    if el is None:
        raise Exception("Dossier รายงานรายจังหวัด introuvable")
    print(f"     text='{el.text[:40]}'")
    safe_click(driver, el)
    time.sleep(PAUSE * 2)


# ─── ÉTAPE 4 : Récupérer les 79 fichiers province ─────────────────────────────
def get_file_items(driver):
    print("  [4] Récupération liste provinces...")
    time.sleep(2)

    # On cherche les divs avec icône cloud = vrais items fichier
    # D'après la capture : chaque item = div avec icône cloud + texte province
    # Ils ont tous le même niveau dans right-card-download
    strategies = [
        # Items avec icône cloud (img ou i ou svg)
        ("XPATH", "//div[contains(@class,'right-card-download')]"
                  "//div[.//img[contains(@src,'cloud')] or "
                  ".//i[contains(@class,'cloud')] or "
                  ".//v-icon[contains(.,'cloud')]]"),
        # Items qui contiennent exactement 1 texte court + une icône
        ("CSS",   ".right-card-download .v-list-item"),
        ("CSS",   ".right-card-download [class*='item-province']"),
        ("CSS",   ".right-card-download [class*='province']"),
        ("CSS",   ".right-card-download [class*='list-item']"),
        # Chercher par structure : div frère d'une icône cloud
        ("XPATH", "//div[contains(@class,'right-card-download')]"
                  "//div[contains(@class,'v-list-item')]"),
    ]

    for method, sel in strategies:
        try:
            if method == "CSS":
                els = driver.find_elements(By.CSS_SELECTOR, sel)
            else:
                els = driver.find_elements(By.XPATH, sel)

            visible = [e for e in els if e.is_displayed() and e.text.strip()]
            # Filtrer les items parasites
            filtered = [e for e in visible
                        if e.text.strip() not in SKIP_TEXTS
                        and '\n' not in e.text.strip()
                        and len(e.text.strip()) < 60]

            if len(filtered) >= 3:
                # Dédupliquer : garder la DERNIÈRE occurrence par texte
                # (dans Vuetify les doublons DOM, le dernier est souvent le vrai cliquable)
                seen = {}
                for e in filtered:
                    txt = e.text.strip()
                    seen[txt] = e  # écrase → garde le dernier
                deduped = list(seen.values())
                print(f"     ✓ {len(deduped)} provinces uniques (sur {len(filtered)} détectées)")
                for v in deduped[:5]:
                    cls = v.get_attribute("class") or ""
                    print(f"        • {repr(v.text.strip())} | class='{cls[:80]}'")
                if len(deduped) > 5:
                    print(f"        ... +{len(deduped)-5} autres")
                return deduped

        except Exception as ex:
            print(f"     ✗ {sel} : {ex}")

    # ── Fallback : prendre TOUS les divs de right-card et filtrer agressivement ──
    print("  Fallback : scan complet right-card-download...")
    try:
        rc = driver.find_element(By.CSS_SELECTOR, ".right-card-download")
        all_divs = rc.find_elements(By.XPATH, ".//div | .//span | .//li")
        filtered = []
        seen = set()
        for el in all_divs:
            txt = el.text.strip()
            if (el.is_displayed()
                    and txt
                    and txt not in SKIP_TEXTS
                    and '\n' not in txt
                    and 2 < len(txt) < 50
                    and txt not in seen):
                # Vérifier que c'est pas un conteneur (pas de sous-divs avec texte)
                children_text = [c.text.strip() for c in el.find_elements(By.XPATH, "./div | ./span")
                                 if c.text.strip()]
                if not children_text:
                    filtered.append(el)
                    seen.add(txt)

        if filtered:
            print(f"     ✓ Fallback → {len(filtered)} provinces")
            for v in filtered[:5]:
                print(f"        • {repr(v.text.strip())}")
            return filtered
    except Exception as e:
        print(f"     Fallback échoué : {e}")

    raise Exception("Liste de fichiers introuvable")


# ─── ÉTAPE 5 : Sélectionner + cliquer Download ────────────────────────────────
def download_one_file(driver, item_el, index, total):
    label = item_el.text.strip().replace("\n", " ") or f"item_{index}"
    print(f"  [{index:03d}/{total}] {label}", end="  ", flush=True)

    # 1. Clic sur l'item pour le sélectionner (✓ vert apparaît)
    safe_click(driver, item_el)
    time.sleep(PAUSE)

    # 2. Vérifier que le ✓ vert est apparu (optionnel, continue quand même)
    # 3. Chercher et cliquer le bouton ดาวน์โหลด orange
    btn = None
    deadline = time.time() + 10
    while time.time() < deadline and btn is None:
        for xpath in [
            # Bouton orange avec texte ดาวน์โหลด, pas disabled
            "//button[contains(@class,'v-btn')][contains(.,'ดาวน์โหลด')][not(@disabled)]",
            "//button[contains(.,'ดาวน์โหลด')][not(@disabled)]",
            # Par style orange rgb(224, 184, 126)
            "//button[contains(@style,'224, 184, 126')]",
            "//button[contains(@style,'224,184,126')]",
            # Dernier recours : tout bouton v-btn visible qui n'est pas le ×
            "//div[contains(@class,'right-card-download')]//button[contains(@class,'v-btn')]",
        ]:
            try:
                for c in driver.find_elements(By.XPATH, xpath):
                    if c.is_displayed() and 'ดาวน์โหลด' in c.text:
                        disabled = c.get_attribute("disabled")
                        if not disabled:
                            btn = c
                            break
            except Exception:
                pass
        if btn is None:
            time.sleep(0.5)

    if btn is None:
        print("✗ bouton ดาวน์โหลด introuvable")
        return False

    print(f"[btn trouvé: '{btn.text[:20]}']", end="  ", flush=True)
    safe_click(driver, btn)
    time.sleep(DOWNLOAD_WAIT)
    print("✓")

    # 4. Désélectionner pour remettre à zéro
    try:
        safe_click(driver, item_el)
        time.sleep(0.5)
    except Exception:
        pass

    return True


# ─── PIPELINE ──────────────────────────────────────────────────────────────────
def process_culture(driver, culture_name: str, log: list):
    print(f"\n{'='*60}")
    print(f"  Culture : {culture_name}")
    print(f"{'='*60}")

    open_download_menu(driver)
    select_culture(driver, culture_name)
    open_xlsx_folder(driver)
    items = get_file_items(driver)

    ok_count = 0
    for i, item in enumerate(items, 1):
        try:
            success = download_one_file(driver, item, i, len(items))
            status = "OK" if success else "SKIP"
            if success:
                ok_count += 1
        except Exception as e:
            print(f"✗ ({e})")
            status = f"ERREUR: {e}"
        log.append({
            "culture": culture_name,
            "fichier": item.text.strip()[:60] if item.text else f"item_{i}",
            "status": status,
        })

    print(f"\n  Sous-total {culture_name} : {ok_count}/{len(items)}")
    close_modal(driver)


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Download dir : {DOWNLOAD_DIR}")
    driver = make_driver(DOWNLOAD_DIR)
    log = []

    try:
        driver.get(URL)
        print("Chargement page (5s)...")
        time.sleep(5)

        for culture in CULTURES:
            try:
                process_culture(driver, culture, log)
            except Exception as e:
                print(f"\n⚠️  Erreur sur {culture} : {e}")
                log.append({"culture": culture, "fichier": "N/A",
                            "status": f"ERREUR CULTURE: {e}"})
                try:
                    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
                    wait_overlay_gone(driver)
                except Exception:
                    pass

    finally:
        driver.quit()

    print(f"\n{'='*60}")
    print("  RÉSUMÉ FINAL")
    print(f"{'='*60}")
    ok  = sum(1 for r in log if r["status"] == "OK")
    err = len(log) - ok
    print(f"  ✓ Succès  : {ok}")
    print(f"  ✗ Erreurs : {err}")
    print(f"  Fichiers dans : {DOWNLOAD_DIR}")
    if err:
        print("\n  Échecs :")
        for r in log:
            if r["status"] != "OK":
                print(f"    [{r['culture']}] {r['fichier']} → {r['status']}")


if __name__ == "__main__":
    main()