import torch
import cv2
import numpy as np
from skimage.measure import regionprops
import pandas as pd
import math
import os
import sys
from cellpose import models

# ================================================================
# 1) CHARGEMENT DU MODÈLE CELLPOSE
# ================================================================
def load_model():
    print("[INFO] Chargement du modèle Cellpose (cyto)...")
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f"[INFO] Device: {device}")
    
    try:
        model = models.Cellpose(gpu=use_gpu, model_type='cyto')
    except AttributeError:
        # Compatibility with older/newer versions
        print("[INFO] Fallback to CellposeModel")
        model = models.CellposeModel(gpu=use_gpu, model_type='cyto')
    return model

# ================================================================
# 2) CLASSIFICATION CELLULAIRE (RBC vs WBC)
# ================================================================
def classify_cell_type(props, pixel_to_micron):
    """
    Classifie une cellule comme 'RBC' ou 'WBC' basé sur sa taille et forme.
    Heuristique simple : 
    - WBC sont généralement plus grands (> 80 µm²) et peuvent être plus irréguliers.
    - RBC typiques ~ 40-50 µm² (normocytes).
    """
    area_um2 = props.area * (pixel_to_micron ** 2)
    
    # Seuil de taille : > 80 µm² -> Probablement un WBC (ou un amas)
    # Dans un frottis sanguin standard, les lymphocytes font ~7-10µm (proche RBC)
    # mais les granulocytes/monocytes sont plus grands (10-20µm).
    # On utilise un seuil prudent pour isoler les "grosses" cellules comme WBC.
    if area_um2 > 80:
        return "WBC"
    else:
        return "RBC"

# ================================================================
# 3) SEGMENTATION ET ANALYSE
# ================================================================
def segment_and_analyze(model, img_input, pixel_to_micron=0.25):
    print(f"[INFO] Traitement de l'image (Input type: {type(img_input)})")
    
    if isinstance(img_input, str):
        if not os.path.exists(img_input):
            raise ValueError(f"Image introuvable: {img_input}")
        img = cv2.imread(img_input)
    elif isinstance(img_input, np.ndarray):
        img = img_input.copy()
        # Convert RGB to BGR if needed, but assuming standard BGR for cv2 processing if coming from cv2.
        # If coming from PIL (RGB), we might need to swap. 
        # CAUTION: GUI usually provides RGB (from PIL). 
        # But this function used to convert BGR (cv2 read) to RGB for cellpose.
        # Let's Standardize: Expect BGR here if array (OpenCV standard).
    else:
        raise ValueError("L'entrée doit être un chemin (str) ou une image (np.ndarray BGR)")

    if img is None:
        raise ValueError(f"Impossible de lire l'image.")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Segmentation Cellpose
    print("[INFO] Segmentation Cellpose en cours...")
    # diameter=30 pixels ~ 7.5um at 0.25um/pix? 
    # 7.5um / 0.25 = 30 pixels. C'est une bonne estimation pour RBC.
    try:
        masks, flows, styles, diams = model.eval(img_rgb, diameter=30, channels=[0, 0])
    except ValueError:
        masks, flows, styles = model.eval(img_rgb, diameter=30, channels=[0, 0])
    
    num_cells = masks.max()
    print(f"[INFO] {num_cells} objets détectés")

    # 2. Analyse individuelle
    rbc_features = []
    classified_cells = [] # Liste de tuples (mask_id, type, contour)

    for i in range(1, num_cells + 1):
        # Isoler le masque binaire
        mask_binary = (masks == i).astype(np.uint8)
        if mask_binary.sum() == 0: continue
        
        props = regionprops(mask_binary)[0]
        
        # Classification
        cell_type = classify_cell_type(props, pixel_to_micron)
        classified_cells.append({
            "id": i,
            "type": cell_type,
            "mask": mask_binary,
            "convex_image": props.convex_image # Pour visualisation potentielle
        })

        # Si RBC, extraction features détaillées
        if cell_type == "RBC":
            area = props.area * (pixel_to_micron ** 2)
            perimeter = props.perimeter * pixel_to_micron
            diameter = props.equivalent_diameter * pixel_to_micron
            major = props.major_axis_length * pixel_to_micron
            minor = props.minor_axis_length * pixel_to_micron
            
            # Feature: Circularity
            # C = 4*pi*A / P^2.  C=1 pour cercle parfait.
            circularity = (4 * math.pi * area) / (perimeter ** 2 + 1e-8)
            
            # Feature: Aspect Ratio & Elongation
            # Aspect Ratio = Major / Minor
            # Elongation = (Major - Minor) / Major
            aspect_ratio = major / (minor + 1e-8)
            elongation = (major - minor) / (major + 1e-8)
            
            # Detection Schistocyte (Fragmenté/Déformé)
            # Critères typiques: Circularité faible (< 0.6) ou Elongation forte (> 0.5 pour elliptocytes/schistos)
            # Schistocytes sont souvent triangulaires ou très irréguliers -> circularité faible.
            is_schistocyte = 1 if (circularity < 0.60) else 0
            
            rbc_features.append({
                "cell_id": i,
                "area_um2": area,
                "perimeter_um": perimeter,
                "diameter_um": diameter,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
                "elongation_index": elongation,
                "is_schistocyte": is_schistocyte
            })

    df_rbc = pd.DataFrame(rbc_features)
    print(f"[INFO] {len(df_rbc)} RBCs identifiés sur {num_cells} objets.")
    
    return img, classified_cells, df_rbc

# ================================================================
# 4) INTERPRÉTATION CLINIQUE
# ================================================================
def interpret_anemia(df_rbc):
    print("[INFO] Interprétation clinique...")
    
    if df_rbc.empty:
        return "Données insuffisantes (Aucun RBC détecté)"
        
    # Calcul MCV Morphologique (Mean Corpuscular Volume)
    # Approx: Volume = Aire * Epaisseur (supposée ~2um)
    # ou simplement utiliser l'aire comme proxy pour micro/macro
    # Normale: 80-100 fL.
    # Ici: Volume ≈ Area * 2.0
    df_rbc["volume_est_fL"] = df_rbc["area_um2"] * 2.0
    
    mcv = df_rbc["volume_est_fL"].mean()
    mcv_std = df_rbc["volume_est_fL"].std()
    
    # Taux de fragmentation
    schisto_count = df_rbc["is_schistocyte"].sum()
    total_rbc = len(df_rbc)
    frag_rate = (schisto_count / total_rbc) * 100
    
    # Classification
    interpretation = []
    diagnosis = "Normal"
    
    print(f"\n--- PARAMÈTRES GLOBAUX ---")
    print(f"MCV Estimé : {mcv:.2f} fL (Norme: 80-100)")
    print(f"Dispersion (RDW-like) : {mcv_std:.2f}")
    print(f"Taux de Schistocytes : {frag_rate:.2f}% (Seuil alerte: >1%)")
    
    if mcv < 80:
        interpretation.append("Microcytose (taille réduite)")
        diagnosis = "Anémie Ferriprive (Probable)"
    elif mcv > 100:
        interpretation.append("Macrocytose (taille augmentée)")
        diagnosis = "Anémie Mégaloblastique (Probable)"
        
    if frag_rate > 1:
        interpretation.append("Présence significative de Schistocytes")
        if diagnosis == "Normal":
            diagnosis = "Anémie Hémolytique (Suspectée)"
        else:
            diagnosis += " / Composante Hémolytique"
            
    if diagnosis == "Normal" and not interpretation:
        diagnosis = "Pas d'anomalie majeure détectée"
        
    print(f"\n>>> DIAGNOSTIC SUGGÉRÉ : {diagnosis}")
    
    return diagnosis, mcv, frag_rate

# ================================================================
# 5) VISUALISATION
# ================================================================
def visualize_results(img, classified_cells, save_path="segmentation_annotated.png"):
    print("[INFO] Génération de la visualisation...")
    viz_img = img.copy()
    
    # Couleurs (BGR)
    COLOR_RBC = (0, 0, 255)   # Rouge
    COLOR_WBC = (255, 0, 0)   # Bleu
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for cell in classified_cells:
        mask = cell["mask"]
        c_type = cell["type"]
        
        # Trouver contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        color = COLOR_RBC if c_type == "RBC" else COLOR_WBC
        
        # Dessiner contour
        cv2.drawContours(viz_img, contours, -1, color, 2)
        
        # Label (optionnel pour ne pas surcharger)
        # On calcule le centre pour placer le label
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            label = "W" if c_type == "WBC" else ""
            if label:
                cv2.putText(viz_img, label, (cX-5, cY+5), font, 0.5, color, 2)

    if save_path:
        cv2.imwrite(save_path, viz_img)
        print(f"[INFO] Image sauvegardée : {save_path}")
        
    return viz_img

# ================================================================
# 6) MAIN
# ================================================================
def analyze_blood_image(input_data):
    # 1. Load
    model = load_model()
    
    # 2. Segment & Analyze
    img, classified_cells, df_rbc = segment_and_analyze(model, input_data)
    
    # 3. Visualize
    viz_img = visualize_results(img, classified_cells, save_path="segmentation_result.png")
    
    # 4. Interpret
    anemia_diag, mcv, frag_rate = interpret_anemia(df_rbc)
    
    # 5. Save Data
    df_rbc.to_csv("rbc_features.csv", index=False)
    print("[INFO] Données exportées dans 'rbc_features.csv'")
    
    return anemia_diag, viz_img

if __name__ == "__main__":
    # Chemin dynamique (comme fixé précédemment)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Ajustez ce chemin si nécessaire
    image_path = os.path.join(script_dir, "redimensionnement", "Original_images", "Original_images_anemic", "001_a.png")
    
    if not os.path.exists(image_path):
        print(f"[ERREUR] Image non trouvée : {image_path}")
    else:
        analyze_blood_image(image_path)
