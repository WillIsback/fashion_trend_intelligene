
from .utils import get_logger
import json
import numpy as np
from .config import CLASS_MAPPING

logger = get_logger(__name__, 'report.log')

def clean_numpy_for_json(obj):
    """Convertit récursivement les types NumPy en types Python"""
    if isinstance(obj, dict):
        return {k: clean_numpy_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_numpy_for_json(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj) if not np.isnan(obj) else None
    elif obj is np.nan:
        return None
    return obj
   
def analyse_evaluation_image(mean_iou, iou_scores, accuracy, distributions_GT, distributions_Pred):
    # Analyse Pixel Accuracy : 
    logger.info("\n=== Analyse Pixel Accuracy ===")
    if accuracy < 70:
        logger.warning(f"Attention la précision du model est faible (inférieure à 70%): {accuracy:.2f}%")
    elif (accuracy > 70 and accuracy < 85):
        logger.info(f"Perfomance de précision du model correcte: {accuracy:.2f}%")
    elif (accuracy > 85 and accuracy < 95):
        logger.info(f"Perfomance de précision du model bonne: {accuracy:.2f}%")
    else:
        logger.info(f"Perfomance de précision du model excellente: {accuracy:.2f}%")
        
    # Analyse Mean IoU
    logger.info("\n=== Analyse Mean IoU ===")
    logger.warning("*Rappel des biais probables* :")
    logger.warning("    IoU faible + Petite classe = Problème de détection")
    logger.warning("    IoU faible + Grande classe = Problème de segmentation")
    logger.info(f"Mean IoU: {mean_iou:.4f}")
    iou_scores_filtered = [iou for iou in iou_scores if iou['class_name'] != 'Background']
    for iou_score in iou_scores_filtered:
        iou = iou_score['iou']
        class_name = iou_score['class_name']
        if iou < 0.5:
            logger.warning(f"Faible IoU pour la classe '{class_name}': {iou:.4f}")
        elif iou >= 0.5 and iou < 0.75:
            logger.info(f"IoU modérée pour la classe '{class_name}': {iou:.4f}")
        else:
            logger.info(f"Bonne IoU pour la classe '{class_name}': {iou:.4f}")
    
    # Analyse distribution des classes + Performance
    logger.info("\n=== Analyse distribution des classes + Performance ===")
    for dist in distributions_GT:
        bad_score = next((iou for iou in iou_scores if iou['class_name'] == dist['class_name'] and iou['iou'] < 0.5), None)
        if bad_score and dist['percentage'] < 1.0:
            logger.warning(f"Classe '{dist['class_name']}' rare en GT ({dist['percentage']:.2f}%) avec faible IoU ({bad_score['iou']:.4f}). Possible problème de détection.")
            logger.warning(f"Détection de cette classe en Prédiction : {next((d['percentage'] for d in distributions_Pred if d['class_name'] == dist['class_name']), 0.0):.2f}%")
        elif bad_score and dist['percentage'] >= 1.0:
            logger.warning(f"Classe '{dist['class_name']}' fréquente en GT ({dist['percentage']:.2f}%) mais faible IoU ({bad_score['iou']:.4f}). Possible problème de segmentation.")
            logger.warning(f"Détection de cette classe en Prédiction : {next((d['percentage'] for d in distributions_Pred if d['class_name'] == dist['class_name']), 0.0):.2f}%")
        
    # Compter les classes par performance
    excellent_classes = len([iou for iou in iou_scores if iou['iou'] >= 0.9])
    good_classes = len([iou for iou in iou_scores if 0.75 <= iou['iou'] < 0.9])
    moderate_classes = len([iou for iou in iou_scores if 0.5 <= iou['iou'] < 0.75])
    poor_classes = len([iou for iou in iou_scores if iou['iou'] < 0.5])

    logger.info(f"Répartition des performances : {excellent_classes} excellentes, {good_classes} bonnes, {moderate_classes} modérées, {poor_classes} faibles")
    

def analyze_dataset_eval(dataset_eval):
    """
    Analyse les résultats d'évaluation pour un ensemble d'images.
    """
    # Stabilité - Écart-type des Mean IoU
    mean_ious = [img['mean_iou'] for img in dataset_eval]
    # Calculs statistiques
    global_mean = np.mean(mean_ious)
    std_iou = np.std(mean_ious)
    # Images problématiques
    poor_images = [img for img in dataset_eval if img['mean_iou'] < 0.5]
    print(f"Nombre d'images avec Mean IoU < 0.5 : {len(poor_images)}")
    
    # Analyse par classe - Stabilité par classe
    class_names = list(CLASS_MAPPING.keys())
    class_names.remove('Background')  # Exclure le fond

    class_iou_dict = {class_name: [] for class_name in class_names}
    for img in dataset_eval:
        for class_name in class_names:
            class_iou = next((score['iou'] for score in img['iou_scores'] if score['class_name'] == class_name), None)
            if class_iou is not None:
                class_iou_dict[class_name].append(class_iou)
                
    class_stability = {}
    for class_name, ious in class_iou_dict.items():
        # Filtrer les NaN AVANT le calcul
        valid_ious = [iou for iou in ious if not np.isnan(iou)]
        
        if len(valid_ious) > 0:  # Au moins une valeur valide
            class_stability[class_name] = {
                'mean_iou': np.mean(valid_ious),
                'std_iou': np.std(valid_ious) if len(valid_ious) > 1 else 0.0
            }
    
    # Analyse globale de la Pixel Accuracy du dataset
    global_pixel_accuracy = np.mean([img['accuracy'] for img in dataset_eval])
    
    # Compter présence par classe
    class_frequency = {}
    for img in dataset_eval:
        for dist in img['distributions_GT']:
            class_name = dist['class_name']
            if class_name not in class_frequency:
                class_frequency[class_name] = 0
            class_frequency[class_name] += 1
            
    # Tri par performance
    sorted_by_iou = sorted(dataset_eval, key=lambda x: x['mean_iou'])
    worst_5 = sorted_by_iou[:5]
    best_5 = sorted_by_iou[-5:]
    print("\n5 images avec les pires Mean IoU :")
    for img in worst_5:
        print(f"Image: {img['image']}, Mean IoU: {img['mean_iou']:.4f}")
    print("\n5 images avec les meilleures Mean IoU :")
    for img in best_5:
        print(f"Image: {img['image']}, Mean IoU: {img['mean_iou']:.4f}")
    
    # Pour chaque classe, compter les échecs
    problematic_classes = {}
    for class_name in class_names:
        failure_count = 0
        total_appearances = 0
        for img in dataset_eval:
            class_iou = next((score['iou'] for score in img['iou_scores'] if score['class_name'] == class_name), None)
            if class_iou is not None:
                total_appearances += 1
                if class_iou < 0.5:
                    failure_count += 1
        failure_rate = failure_count / total_appearances if total_appearances > 0 else 0
        if failure_rate > 0.5:  # Problématique dans >50% des cas
            problematic_classes[class_name] = failure_rate
            
    dataset_eval = clean_numpy_for_json(dataset_eval)
    dataset_results = {
        'global_metrics': {
            'mean_iou': float(global_mean), 
            'pixel_accuracy': float(global_pixel_accuracy)
            },
        'stability_metrics': {
            'std_iou': float(std_iou), 
            'class_stability': {
                class_name: {
                        'mean_iou': float(stats['mean_iou']) if not np.isnan(stats['mean_iou']) else None,
                        'std_iou': float(stats['std_iou']) if not np.isnan(stats['std_iou']) else None
                    } for class_name, stats in class_stability.items()
               }
            },
        'class_analysis': problematic_classes,
        'per_image_results': dataset_eval
    }
    dataset_results = {
        'global_metrics': {
            'mean_iou': float(global_mean), 
            'pixel_accuracy': float(global_pixel_accuracy),
            'total_images': len(dataset_eval)
        },
        'stability_metrics': {
            'std_iou': float(std_iou), 
            'class_stability': {
                class_name: {
                        'mean_iou': float(stats['mean_iou']) if not np.isnan(stats['mean_iou']) else None,
                        'std_iou': float(stats['std_iou']) if not np.isnan(stats['std_iou']) else None
                    } for class_name, stats in class_stability.items()
               }
        },
        'class_frequency': {
            class_name: freq for class_name, freq in class_frequency.items()
        },
        'problematic_classes': problematic_classes,
        'performance_ranking': {
            'worst_5': [{'image': img['image'], 'mean_iou': float(img['mean_iou'])} for img in worst_5],
            'best_5': [{'image': img['image'], 'mean_iou': float(img['mean_iou'])} for img in best_5]
        },
        'per_image_results': clean_numpy_for_json(dataset_eval)
    }
    # ecritue du json complet des resultats
    with open('dataset_evaluation_report.json', 'w') as f:
        json.dump(dataset_results, f, indent=4)
        logger.info("Rapport d'évaluation du dataset sauvegardé dans 'dataset_evaluation_report.json'")
    
    return dataset_results
