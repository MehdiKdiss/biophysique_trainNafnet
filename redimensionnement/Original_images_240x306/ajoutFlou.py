import cv2
import numpy as np
import os


def ajouter_flou(image, k=5):
    return cv2.GaussianBlur(image, (k, k), 0)

def ajouter_flou_dossier(input_folder, output_folder):
    print(f"\n--- TRAITEMENT DU DOSSIER ---")
    print(f"Input : {input_folder}")
    
    if not os.path.exists(input_folder):
        print("Le dossier d'entrée n'existe pas !")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Dossier créé : {output_folder}")

    files = os.listdir(input_folder)
    print(f"Fichiers trouvés : {files}")

    count = 0

    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

            img_path = os.path.join(input_folder, filename)
            print(f"Traitement : {img_path}")

            img = cv2.imread(img_path)

            if img is None:
                print(f"Impossible de lire : {filename}")
                continue

            img_floue = ajouter_flou(img)

            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, img_floue)
            count += 1

    print(f"Terminé : {count} images enregistrées dans {output_folder}")

# ------------------------------------------------------
# CHEMINS
# ------------------------------------------------------
base_path = r"c:\Users\Asus\Desktop\projet 2025-2026\biophysique\Original_images_240x306"

dossier_anemic = os.path.join(base_path, "anemic")
dossier_healthy = os.path.join(base_path, "healthy")

dossier_anemic_flou = os.path.join(base_path, "anemic_flou")
dossier_healthy_flou = os.path.join(base_path, "healthy_flou")

ajouter_flou_dossier(dossier_anemic, dossier_anemic_flou)
ajouter_flou_dossier(dossier_healthy, dossier_healthy_flou)
