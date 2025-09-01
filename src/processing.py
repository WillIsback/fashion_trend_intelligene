from tqdm import tqdm
import os
import time 
from .utils import get_image_dimensions, create_masks, get_logger, save_results
from .api import call_hf_segmentation_api
import numpy as np
import matplotlib as mplt
from .config import API_SEGMENTATION_OUTPUTS_DIR, WWG_SEGMENTATION_OUTPUTS_DIR
import cv2

logger = get_logger(__name__, __name__ + ".log")

mplt.use('Agg')  # Pour les environnements sans interface graphique 

def segment_images_batch(list_of_image_paths):
    """
    Segmente une liste d'images en utilisant l'API Hugging Face de manière séquentielle.
    """
   
    for idx, img_path in enumerate(tqdm(list_of_image_paths, desc="Segmentation des images")):
        image_filename = os.path.basename(img_path)
        mask_filename = image_filename.replace("image", "mask").rsplit('.', 1)[0] + ".png"
        print(f"\n--- Traitement de l'image {idx+1}/{len(list_of_image_paths)} ---")
        print(f"Fichier: {image_filename}")
        
        try:
            # Obtenir les dimensions de l'image originale
            width, height = get_image_dimensions(img_path)
            print(f"Dimensions: {width}x{height}")
            
            # Appeler l'API avec le chemin de fichier (méthode qui fonctionne)
            print("Envoi de la requête à l'API...")
            start_time = time.time()
            
            output = call_hf_segmentation_api(img_path)
            
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"Réponse reçue en {processing_time:.2f} secondes")
            logger.info(f"Image: {os.path.basename(img_path)} - Processing Time: {processing_time:.2f} seconds")
            
            # Vérifier la structure de la réponse
            if isinstance(output, list) and len(output) > 0:
                print(f"Nombre de segments détectés: {len(output)}")
                
                # Créer le masque de segmentation combiné avec palette et labels
                print("Création du masque de segmentation...")
                segmentation_result = create_masks(output, width, height)
                combined_mask = segmentation_result
                
                # Statistiques du masque
                unique_classes = np.unique(combined_mask)
                print(f"Classes présentes: {unique_classes}")
                
                
                print("Segmentation terminée avec succès")
                logger.info(f"Image: {os.path.basename(img_path)} - Classes Detected: {unique_classes}")
                
                # Sauvegarder les résultats dans un répertoire spécifique
                if not os.path.exists(API_SEGMENTATION_OUTPUTS_DIR):
                    os.makedirs(API_SEGMENTATION_OUTPUTS_DIR, exist_ok=True)
                    
                seg_mask_np = np.array(combined_mask)
                # Si le masque possède 3 canaux, le convertir en niveaux de gris
                if seg_mask_np.ndim == 3:
                    seg_mask_np = cv2.cvtColor(seg_mask_np, cv2.COLOR_BGR2GRAY)
                    
                output_mask_path = os.path.join(API_SEGMENTATION_OUTPUTS_DIR, "Mask", mask_filename)
                os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
                output_img_path = os.path.join(API_SEGMENTATION_OUTPUTS_DIR, "IMG", image_filename)
                os.makedirs(os.path.dirname(output_img_path), exist_ok=True)

                cv2.imwrite(output_mask_path, seg_mask_np)
                
                image_np = np.array(cv2.imread(img_path))
                # # Conversion de l'image de RGB à BGR (pour cv2.imwrite)
                # image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(output_img_path, image_np)

                print(f"Sauvegardé: {mask_filename}")
                
            
            else:
                print("Réponse API invalide")

        except Exception as e:
            print(f"Erreur lors du traitement de {img_path}: {e}")
        
        # Temporisation entre les requêtes pour éviter de surcharger l'API
        if idx < len(list_of_image_paths) - 1:  # Pas de pause après la dernière image
            print("Pause de 2 secondes avant la prochaine requête...")
            time.sleep(2)
            
    # Création du visuel de comparaison
    output_mask_dir = os.path.join(API_SEGMENTATION_OUTPUTS_DIR, "Mask")
    output_img_dir = os.path.join(API_SEGMENTATION_OUTPUTS_DIR, "IMG")
    save_results(output_img_dir, output_mask_dir, WWG_SEGMENTATION_OUTPUTS_DIR)
    




# def save_segmented_images_batch(original_image_paths, segmentation_results, output_dir="output_segmentation"):
#     print(f"\n Sauvegarde des résultats dans {output_dir}/...")
#     os.makedirs(output_dir, exist_ok=True)
    
#     saved_count = 0
#     for idx, (img_path, result) in enumerate(zip(original_image_paths, segmentation_results)):
#         if result is None:
#             print(f"Ignorer l'image {idx+1} (erreur de segmentation)")
#             continue
            
#         seg_mask, color_palette, detected_labels = result
        
#         try:
#             fig, axes = plt.subplots(1, 2, figsize=(14, 7))
#             cmap = mcolors.ListedColormap(color_palette)
            
#             # Image originale
#             original_img = Image.open(img_path)
#             axes[0].imshow(original_img)
#             axes[0].set_title(f"Image Originale\n{os.path.basename(img_path)}")
#             axes[0].axis('off')

#             # Masque segmenté superposé sur l'image originale
#             axes[1].imshow(original_img)
#             axes[1].imshow(seg_mask, cmap=cmap, interpolation='nearest', alpha=0.6)
#             axes[1].set_title("Masque Segmenté")
#             axes[1].axis('off')
            
#             # Créer la légende avec des carrés colorés 
#             legend_elements = []
#             for class_id, label in detected_labels.items():
#                 color = color_palette[class_id]
#                 legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=label))
            
#             # Ajouter la légende
#             if legend_elements:
#                 axes[1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), 
#                              fontsize=10, frameon=True, fancybox=True, shadow=True)
                        
#             filename = f"segmentation_{idx+1}_{os.path.splitext(os.path.basename(img_path))[0]}.png"
#             output_path = os.path.join(output_dir, filename)
#             plt.savefig(output_path, dpi=150, bbox_inches='tight')
#             plt.close(fig)
#             print(f"Sauvegardé: {filename}")
#             saved_count += 1
#         except Exception as e:
#             print(f"Erreur lors de la sauvegarde de l'image {idx+1}: {e}")
    
#     print(f"{saved_count} résultat(s) sauvegardé(s) dans {output_dir}/")