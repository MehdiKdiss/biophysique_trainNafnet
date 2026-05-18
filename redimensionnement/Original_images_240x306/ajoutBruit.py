import cv2
import numpy as np
import os

# ------------------------------------------------------
# Fonction : Ajouter un bruit gaussien
# ------------------------------------------------------
def bruit_gaussien(image, mean=0, sigma=20):
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    image_bruitee = image.astype(np.float32) + gauss
    image_bruitee = np.clip(image_bruitee, 0, 255)
    return image_bruitee.astype(np.uint8)

# ------------------------------------------------------
# Fonction : traiter un dossier
# ------------------------------------------------------
def ajouter_bruit_dossier(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Impossible de lire : {filename}")
                continue

            img_bruitee = bruit_gaussien(img)

            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, img_bruitee)

    print(f"Traitement terminé : {output_folder}")

# ------------------------------------------------------
# MAIN : traitement selon tes chemins
# ------------------------------------------------------
base_path = r"c:\Users\Asus\Desktop\projet 2025-2026\biophysique\Original_images_240x306"

dossier_anemic= os.path.join(base_path, "anemic")
dossier_healthy= os.path.join(base_path, "healthy")

dossier_anemic_bruit = os.path.join(base_path, "sujet anemic_bruit")
dossier_healthy_bruit= os.path.join(base_path, "sujet healthy_bruit")

ajouter_bruit_dossier(dossier_anemic, dossier_anemic_bruit)
ajouter_bruit_dossier(dossier_healthy, dossier_healthy_bruit)

