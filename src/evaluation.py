from .config import EXPECTED_SEGMENTATION_OUTPUTS_DIR, IMG_DIR, MASK_DIR
from .utils import save_results, get_logger
import os
from sklearn.metrics import jaccard_score
import numpy as np
import cv2
from .config import CLASS_MAPPING


logger = get_logger(__name__, __name__ + ".log")


def save_visual_expected_result():
    """
    Sauvegarde des images avec overlay mask attendus. A faire qu'une seule fois pour tout un jeu de donnée avec mask de segmentation déjà fourni.
    """
    print("\nSauvegarde des images attendus ..")
    # Verifier si EXPECTED_SEGMENTATION_OUTPUTS_DIR existe est déjà remplie
    if not os.path.exists(EXPECTED_SEGMENTATION_OUTPUTS_DIR) or not os.listdir(EXPECTED_SEGMENTATION_OUTPUTS_DIR):
        # Sauvegarde d'image comparative des résultats attendus
        save_results(IMG_DIR, MASK_DIR, EXPECTED_SEGMENTATION_OUTPUTS_DIR)


""" 
Métriques Primaires :
    - Mean IoU (global et per-class)
    - Dice Score (pour classes déséquilibrées)
    - Boundary IoU (qualité des contours)
Métriques Secondaires :
    - Pixel Accuracy (performance globale)
    - Mean Surface Distance (précision géométrique)
    - Fashion-specific confusion matrix (erreurs sémantiques)
Métriques de Validation :
    - Cross-validation IoU (stabilité)
    - Inference time (applicabilité pratique)
    - Hierarchical accuracy (cohérence mode)
 """

PRIMARY_METRICS = ["Mean_IoU", "F1-Score", "Boundary_IoU"]

SECONDARY_METRICS = ["Pixel_Accuracy", "Mean_Surface_Distance", "Fashion_specific_confusion_matrix"]

VALIDATION_METRICS = ["Cross_validation_IoU", "Inference_Time", "Hierarchical_accuracy"]

MASK_PRED_DIR = "content/top_influenceurs_2024/Output_API/Mask"
MASK_TRUE_DIR = "content/top_influenceurs_2024/Mask"

def get_y_from_mask(mask_path):
    """
    Charge un masque de segmentation et retourne un tableau y
    """
    # Verification si le fichier existe
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Le fichier de masque '{mask_path}' est introuvable.")
    # Chargement du masque
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Le masque '{mask_path}' est introuvable ou illisible.")
    # Verification la plage de valeurs du masque    
    if mask.min() < 0 or mask.max() >= len(CLASS_MAPPING):
        raise ValueError(f"Le masque '{mask_path}' contient des valeurs hors de la plage attendue (0-{len(CLASS_MAPPING)-1}).")
    print(np.unique(mask))
    logger.info(f"\n Id classe présente du masque '{mask_path}': {np.unique(mask)}")
    return mask


def calculate_mean_iou(y_true, y_pred):
    """
    Calcule Mean IoU entre masques ground truth et prédits
    """
    logger.info("\n=== Calcul Mean IoU ===")
    num_classes = len(CLASS_MAPPING)
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    # Masque des pixels non-Background
    non_bg_mask = (y_true_flat != 0)
    # Filtrer les deux tableaux
    y_true_filtered = y_true_flat[non_bg_mask]
    y_pred_filtered = y_pred_flat[non_bg_mask]
    
    iou_scores = []
    
    class_names = list(CLASS_MAPPING.keys())
    warning_messages = []
    
    for class_id in range(num_classes):
        true_binary = (y_true_filtered == class_id)
        pred_binary = (y_pred_filtered == class_id)
        
        if true_binary.sum() == 0 and pred_binary.sum() == 0: # Si la classe est absente dans GT et prédiction,
            iou = 1.0
        elif true_binary.sum() == 0: # Si la classe est absente dans GT mais présente dans prédiction (Peut être un faux positif),
            iou = np.nan
            logger.warning(f"Classe '{class_names[class_id]}' absente en ground truth. IoU non défini.")
        else: # Calcul IoU normalement
            iou = jaccard_score(true_binary, pred_binary, average='binary')
        
        iou_score= {
            'class_name': class_names[class_id],
            'iou': iou
        }
        
        iou_scores.append(iou_score)
        print(f"IoU for {class_names[class_id]}: {iou:.4f}")
        logger.info(f"IoU for {class_names[class_id]}: {iou:.4f}")
        if iou < 0.5:
            warning_message = {
                'class_id': class_id,
                'mess': f"Faible IoU pour la classe '{class_names[class_id]}': {iou:.4f}"
            }
            warning_messages.append(warning_message)

    mean_iou = np.nanmean([score['iou'] for score in iou_scores])
    print(f"Mean IoU: {mean_iou:.4f}")
    logger.info(f"Mean IoU: {mean_iou:.4f}")
    
    return mean_iou, iou_scores, warning_messages

def analyze_class_distribution(y_true, y_pred, warning_messages=None):
    """Analyse la distribution des classes pour identifier les déséquilibres"""
    print("\n=== Analyse de distribution ===")
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    
    print("Distribution Ground Truth:")
    logger.info("Distribution Ground Truth:")
    distributions_GT = []
    for cls, count in zip(unique_true, counts_true):
        if cls < len(CLASS_MAPPING):
            class_name = list(CLASS_MAPPING.keys())[cls]
            print(f"  {class_name} ({cls}): {count} pixels ({count/y_true.size*100:.1f}%)")
            logger.info(f"  {class_name} ({cls}): {count} pixels ({count/y_true.size*100:.1f}%)")
            distribution_GT={
                'class_name': class_name,
                'class_id': cls,
                'count': count,
                'percentage': count/y_true.size*100
            }
            distributions_GT.append(distribution_GT)
            # Augmentation du message d'avertissement si la classe est déjà signalée
            if(warning_messages is not None):
                warning_found = next((msg for msg in warning_messages if msg['class_id'] == cls), None)
                if warning_found:
                    warning_found['mess'] += f" | Ground Truth: {count} pixels ({count/y_true.size*100:.1f}%)"
                
    print("\nDistribution Prédiction:")
    logger.info("Distribution Prédiction:")
    distributions_Pred = []
    for cls, count in zip(unique_pred, counts_pred):
        if cls < len(CLASS_MAPPING):
            class_name = list(CLASS_MAPPING.keys())[cls]
            print(f"  {class_name} ({cls}): {count} pixels ({count/y_pred.size*100:.1f}%)")
            logger.info(f"  {class_name} ({cls}): {count} pixels ({count/y_pred.size*100:.1f}%)")
            distribution_Pred= {
                'class_name': class_name,
                'class_id': cls,
                'count': count,
                'percentage': count/y_pred.size*100 
            }
            distributions_Pred.append(distribution_Pred)
            # Augmentation du message d'avertissement si la classe est déjà signalée
            if(warning_messages is not None):
                warning_found = next((msg for msg in warning_messages if msg['class_id'] == cls), None)
                if warning_found:
                    warning_found['mess'] += f" | Predicted: {count} pixels ({count/y_true.size*100:.1f}%)"
        else:
            print(f"  CLASSE INCONNUE ({cls}): {count} pixels")
            logger.warning(f"  CLASSE INCONNUE ({cls}): {count} pixels")
            
    return distributions_GT, distributions_Pred


def calculate_pixel_accuracy(y_true, y_pred):
    """
    Pixel Accuracy = (Nombre de pixels correctement prédits) / (Nombre total de pixels) × 100, il mesure le pourcentage de pixels correctements classifiées.
    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
        
    """
    if y_true.shape == y_pred.shape:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten() 
        # Masque des pixels non-Background
        non_bg_mask = (y_true_flat != 0)
        # Filtrer les deux tableaux
        y_true_filtered = y_true_flat[non_bg_mask]
        y_pred_filtered = y_pred_flat[non_bg_mask]
        # Calcul accuracy sans Background
        correct_mask = (y_true_filtered == y_pred_filtered)
        accuracy = np.sum(correct_mask) / len(y_true_filtered) * 100
        print(f"La valeur de la metrique Pixel Accuracy est : {accuracy}")
        logger.info(f"La valeur de la metrique Pixel Accuracy est : {accuracy}")
        return accuracy
    else:
        logger.error("les dimensions du mask prédit et GT ne sont pas égal")
        raise ValueError("les dimensions du mask prédit et GT ne sont pas égal")


                
def evaluate_single_image():
    """
        Calcule et affiche les métriques pour une seule paire de masques (prédit vs ground truth)
    """
    # Selectionne un mask predit random
    random_pred_mask_file = np.random.choice(os.listdir(MASK_PRED_DIR))
    # random_pred_mask_file = "mask_23.png"
    print(f"\nEvaluation du masque prédit: {random_pred_mask_file}")
    logger.info(f"\nEvaluation du masque prédit: {random_pred_mask_file}")
    
    # Recuperation du mask ground truth correspondant
    corresponding_true_mask_file = random_pred_mask_file  # Supposons que les noms de fichiers correspondent
    if not os.path.exists(os.path.join(MASK_TRUE_DIR, corresponding_true_mask_file)):
        raise FileNotFoundError(f"Le masque ground truth '{corresponding_true_mask_file}' est introuvable.")
    print(f"Masque ground truth correspondant: {corresponding_true_mask_file}")
    logger.info(f"Masque ground truth correspondant: {corresponding_true_mask_file}")
    
    y_true = get_y_from_mask(os.path.join(MASK_TRUE_DIR, corresponding_true_mask_file))
    y_pred = get_y_from_mask(os.path.join(MASK_PRED_DIR, random_pred_mask_file))
    
    print("\n==============Mean IoU metrique==============\n")
    logger.info("==============Mean IoU metrique==============")
    mean_iou, iou_scores, warning_messages = calculate_mean_iou(y_true, y_pred)
    distributions_GT, distributions_Pred = analyze_class_distribution(y_true, y_pred, warning_messages=warning_messages)

    print("\n=== Warnings ===")
    for msg in warning_messages:
        logger.warning(msg['mess'])
        print(msg['mess'])
        
    print("\n==============Pixel Accuracy metrique==============\n")
    logger.info("==============Pixel Accuracy metrique==============")
    accuracy = calculate_pixel_accuracy(y_true, y_pred)
    
    return mean_iou, iou_scores, accuracy, distributions_GT, distributions_Pred

    
                
def eval_dataset():
    """
        Calcule et affiche les métriques pour une seule paire de masques (prédit vs ground truth)
    """
    list_metrics_per_img = []
    print("\nEvaluation du jeu de donnée complet...")
    logger.info("\nEvaluation du jeu de donnée complet...")
    
    # Pour chaque mask predit
    for msk in os.listdir(MASK_PRED_DIR):
        # random_pred_mask_file = "mask_23.png"
        print(f"\nEvaluation du masque prédit: {msk}")
        logger.info(f"\nEvaluation du masque prédit: {msk}")
        
        # Recuperation du mask ground truth correspondant
        corresponding_true_mask_file = msk  # Supposons que les noms de fichiers correspondent
        if not os.path.exists(os.path.join(MASK_TRUE_DIR, corresponding_true_mask_file)):
            raise FileNotFoundError(f"Le masque ground truth '{corresponding_true_mask_file}' est introuvable.")
        print(f"Masque ground truth correspondant: {corresponding_true_mask_file}")
        logger.info(f"Masque ground truth correspondant: {corresponding_true_mask_file}")
        
        y_true = get_y_from_mask(os.path.join(MASK_TRUE_DIR, corresponding_true_mask_file))
        y_pred = get_y_from_mask(os.path.join(MASK_PRED_DIR, msk))
        
        print("\n==============Mean IoU metrique==============\n")
        logger.info("==============Mean IoU metrique==============")
        mean_iou, iou_scores, _ = calculate_mean_iou(y_true, y_pred)
        distributions_GT, distributions_Pred = analyze_class_distribution(y_true, y_pred)

            
        print("\n==============Pixel Accuracy metrique==============\n")
        logger.info("==============Pixel Accuracy metrique==============")
        accuracy = calculate_pixel_accuracy(y_true, y_pred)
        
        per_image_results = {
            'image': msk,
            'mean_iou': mean_iou,
            'iou_scores': iou_scores,
            'accuracy': accuracy,
            'distributions_GT': distributions_GT,
            'distributions_Pred': distributions_Pred
        }
        list_metrics_per_img.append(per_image_results)
        
    return list_metrics_per_img
    