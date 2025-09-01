import os
import dotenv
from src.processing import segment_images_batch
from src.evaluation import save_visual_expected_result, evaluate_single_image, eval_dataset
from src.analyzer import analyse_evaluation_image, analyze_dataset_eval
from src.report import fill_template_and_save
import argparse

dotenv.load_dotenv()

if (os.getenv("HF_TOKEN") is None) or (os.getenv("HF_TOKEN") == ""):
    raise ValueError("Vous devez définir la variable d'environnement HF_TOKEN dans le fichier .env")

image_dir = "content/top_influenceurs_2024/IMG"

def main(sample_run=False):
    
    parser = argparse.ArgumentParser(description='Fashion Trend Intelligence')
    
    # Arguments d'options
    parser.add_argument('-s', '--sample',
                        default=False,
                        help='Mode sample pour traiter un petit nombre d\'images (défaut: False)',
                        action='store_true')
    parser.add_argument('-e', '--evaluation', default=False,
                        help="Mode pour faire l'évaluation du modele de segmentation (défaut: False)",
                        action='store_true')

    args = parser.parse_args()
    
    sample_run = args.sample
    eval_mode = args.evaluation
    
    if not os.path.exists(image_dir):
        try:
            os.makedirs(image_dir, exist_ok=True)
            print(f"Répertoire créé : {image_dir}")
        except PermissionError:
            print(f"Permission refusée pour créer le répertoire : {image_dir}")
            print("Veuillez créer le répertoire manuellement ou changer le chemin.")
            return
        
    image_paths = []
    for img in os.listdir(image_dir):
        print(f"Image trouvée : {img}")
        image_paths.append(os.path.join(image_dir, img))
        
    if not image_paths:
        print(f"Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
        return
    else:
        print(f"{len(image_paths)} image(s) à traiter : {image_paths}")

    if not eval_mode:
        print("\nMode segmentation activé...")
        if sample_run:
            image_paths = image_paths[:5]
            print(f"Sample run : {len(image_paths)} image(s) sélectionnée(s) : {image_paths}")
            
        print(f"\nDémarrage du traitement de {len(image_paths)} image(s) en batch...")
        segment_images_batch(image_paths)
        save_visual_expected_result()
    else:
        print("\nMode évaluation activé...")
        mean_iou, iou_scores, accuracy, distributions_GT, distributions_Pred = evaluate_single_image()
        analyse_evaluation_image(mean_iou, iou_scores, accuracy, distributions_GT, distributions_Pred)
        all_img_eval = eval_dataset()
        dataset_results = analyze_dataset_eval(all_img_eval)
        report_path = fill_template_and_save(dataset_results)
        print(f"Rapport complet généré dans : {report_path}")


if __name__ == "__main__":
     
    exit(main())