import matplotlib.pyplot as plt
from pathlib import Path
import shutil


def copy_result_images(performance_ranking, output_dir, real_results_dir="content/top_influenceurs_2024/Real_Results"):
    """Copie les images de résultats dans le dossier du rapport"""
    
    img_dir = output_dir / "img"
    real_results_path = Path(real_results_dir)
    
    image_paths = {}
    
    # Meilleure image
    if performance_ranking['best_5']:
        best_image = performance_ranking['best_5'][-1]['image']  # Dernière = meilleure
        best_number = best_image.replace('mask_', '').replace('.png', '')
        best_result = f"result_{best_number}.png"
        
        source_best = real_results_path / best_result
        dest_best = img_dir / f"best_{best_result}"
        
        if source_best.exists():
            shutil.copy2(source_best, dest_best)
            image_paths['best'] = f"img/{dest_best.name}"
            image_paths['titre_best_image'] = f"Best segmentation (Image {best_number})"
        else:
            image_paths['best'] = "*Image non trouvée*"
            image_paths['titre_best_image'] = ""
    
    # Pire image
    if performance_ranking['worst_5']:
        worst_image = performance_ranking['worst_5'][0]['image']  # Première = pire
        worst_number = worst_image.replace('mask_', '').replace('.png', '')
        worst_result = f"result_{worst_number}.png"
        
        source_worst = real_results_path / worst_result
        dest_worst = img_dir / f"worst_{worst_result}"
        
        if source_worst.exists():
            shutil.copy2(source_worst, dest_worst)
            image_paths['worst'] = f"img/{dest_worst.name}"
            image_paths['titre_worst_image'] = f"Problematic segmentation (Image {worst_number})"
        else:
            image_paths['worst'] = "*Image non trouvée*"
            image_paths['titre_worst_image'] = ""
    
    return image_paths, worst_number

def format_worst_image_analysis(worst_mask_data):
    """Formate l'analyse détaillée de la pire image avec logique complète"""
    if not worst_mask_data:
        return "*Données non disponibles*"
    
    # Extraction des métriques clés
    mean_iou = worst_mask_data['mean_iou']
    accuracy = worst_mask_data['accuracy']
    iou_scores = worst_mask_data['iou_scores']
    distributions_GT = worst_mask_data['distributions_GT']
    distributions_Pred = worst_mask_data['distributions_Pred']
    
    # Construction du texte d'analyse
    analysis = ""
    
    # === Analyse Pixel Accuracy ===
    analysis += "### 📊 Analyse Pixel Accuracy\n\n"
    if accuracy < 70:
        analysis += f"⚠️ **Attention** : Précision faible (< 70%) : {accuracy:.1f}%\n\n"
    elif 70 <= accuracy < 85:
        analysis += f"✅ Performance correcte : {accuracy:.1f}%\n\n"
    elif 85 <= accuracy < 95:
        analysis += f"🟢 **Bonne** performance : {accuracy:.1f}%\n\n"
    else:
        analysis += f"🏆 **Excellente** performance : {accuracy:.1f}%\n\n"
    
    # === Analyse Mean IoU ===
    analysis += "### 🎯 Analyse Mean IoU\n\n"
    analysis += f"**Mean IoU global** : {mean_iou*100:.1f}%\n\n"
    analysis += "*💡 Rappel* : IoU faible + Petite classe = Problème de détection | IoU faible + Grande classe = Problème de segmentation\n\n"
    
    # Filtrer le Background pour l'analyse
    iou_scores_filtered = [iou for iou in iou_scores if iou['class_name'] != 'Background']
    
    # Classes par performance
    analysis += "**Détail par classe** :\n"
    for iou_score in iou_scores_filtered:
        iou = iou_score['iou']
        class_name = iou_score['class_name']
        
        if iou is None:
            analysis += f"- **{class_name}** : Absence totale (0%)\n"
        elif iou < 0.5:
            analysis += f"- 🔴 **{class_name}** : {iou*100:.1f}% (Faible)\n"
        elif 0.5 <= iou < 0.75:
            analysis += f"- 🟡 **{class_name}** : {iou*100:.1f}% (Modérée)\n"
        else:
            analysis += f"- 🟢 **{class_name}** : {iou*100:.1f}% (Bonne)\n"
    
    analysis += "\n"
    
    # === Analyse distribution + Performance ===
    analysis += "### 🔍 Analyse Distribution vs Performance\n\n"
    
    problematic_analysis = []
    for dist in distributions_GT:
        bad_score = next((iou for iou in iou_scores if iou['class_name'] == dist['class_name'] and iou['iou'] is not None and iou['iou'] < 0.5), None)
        pred_percentage = next((d['percentage'] for d in distributions_Pred if d['class_name'] == dist['class_name']), 0.0)
        
        if bad_score and dist['percentage'] < 1.0:
            problematic_analysis.append(f"- 🚨 **{dist['class_name']}** : Classe rare ({dist['percentage']:.1f}% GT) avec IoU faible ({bad_score['iou']*100:.1f}%) → Problème de détection")
            problematic_analysis.append(f"  - Détection en prédiction : {pred_percentage:.1f}%")
        elif bad_score and dist['percentage'] >= 1.0:
            problematic_analysis.append(f"- ⚠️ **{dist['class_name']}** : Classe fréquente ({dist['percentage']:.1f}% GT) mais IoU faible ({bad_score['iou']*100:.1f}%) → Problème de segmentation")
            problematic_analysis.append(f"  - Détection en prédiction : {pred_percentage:.1f}%")
    
    if problematic_analysis:
        analysis += "\n".join(problematic_analysis) + "\n\n"
    else:
        analysis += "*Aucun problème majeur de distribution détecté.*\n\n"
    
    # === Répartition des performances ===
    analysis += "### 📈 Répartition des Performances\n\n"
    
    # Compter les classes par performance (exclure Background et None)
    valid_ious = [iou['iou'] for iou in iou_scores_filtered if iou['iou'] is not None]
    
    excellent_classes = len([iou for iou in valid_ious if iou >= 0.9])
    good_classes = len([iou for iou in valid_ious if 0.75 <= iou < 0.9])
    moderate_classes = len([iou for iou in valid_ious if 0.5 <= iou < 0.75])
    poor_classes = len([iou for iou in valid_ious if iou < 0.5])
    
    total_classes = len(valid_ious)
    
    analysis += f"- 🏆 **Excellentes** (≥90%) : {excellent_classes}/{total_classes} classes\n"
    analysis += f"- 🟢 **Bonnes** (75-90%) : {good_classes}/{total_classes} classes\n"
    analysis += f"- 🟡 **Modérées** (50-75%) : {moderate_classes}/{total_classes} classes\n"
    analysis += f"- 🔴 **Faibles** (<50%) : {poor_classes}/{total_classes} classes\n\n"
    
    
    return analysis

def create_images_table(images_list, title):
    """Crée un tableau pour le top/flop des images"""
    if not images_list:
        return "*Aucune image dans cette catégorie*\n"
    
    table = "| Image | Mean IoU | Performance |\n"
    table += "|-------|----------|-------------|\n"
    
    for img in images_list:
        performance = "🔴 Faible" if img['mean_iou'] < 0.6 else "🟡 Moyenne" if img['mean_iou'] < 0.8 else "🟢 Bonne"
        table += f"| {img['image']} | {img['mean_iou']*100:.1f}% | {performance} |\n"
    
    return table

def generate_stability_analysis(json_data):
    """Génère l'analyse de stabilité"""
    std_iou = json_data['stability_metrics']['std_iou']
    
    if std_iou < 0.05:
        stability_level = "🟢 **Très stable**"
    elif std_iou < 0.1:
        stability_level = "🟡 **Modérément stable**"
    else:
        stability_level = "🔴 **Instable**"
    
    return f"""
Le modèle présente une stabilité {stability_level} avec un écart-type de ±{std_iou*100:.1f}%.

**Interprétation** : Les performances varient en moyenne de ±{std_iou*100:.1f}% entre les images.
"""

def generate_warning_analysis(json_data):
    """Génère l'analyse des classes problématiques"""
    problematic = json_data['problematic_classes']
    
    if not problematic:
        return "*Aucune classe particulièrement problématique détectée.*"
    
    warnings = []
    for class_name, failure_rate in problematic.items():
        if failure_rate > 0.8:
            warnings.append(f"- 🚨 **{class_name}** : {failure_rate*100:.0f}% d'échecs - Révision urgente")
        elif failure_rate > 0.6:
            warnings.append(f"- ⚠️ **{class_name}** : {failure_rate*100:.0f}% d'échecs - Amélioration recommandée")
        else:
            warnings.append(f"- 💡 **{class_name}** : {failure_rate*100:.0f}% d'échecs - Surveillance")
    
    return "\n".join(warnings)


def generate_charts(json_data, img_dir):
    """Génère les graphiques et retourne leurs chemins"""
    
    charts_paths = {}
    
    # 1. Chart des performances par classe
    charts_paths['performance'] = create_performance_chart(json_data, img_dir)
    
    # 2. Chart de stabilité
    charts_paths['stability'] = create_stability_chart(json_data, img_dir)
    
    # 3. Chart de fréquence des classes
    charts_paths['frequency'] = create_frequency_chart(json_data, img_dir)
    
    return charts_paths

def create_performance_chart(json_data, img_dir):
    """Crée un bar chart des performances par classe"""
    class_stability = json_data['stability_metrics']['class_stability']
    
    classes = list(class_stability.keys())
    mean_ious = [stats['mean_iou'] * 100 for stats in class_stability.values()]
    colors = ['green' if iou >= 90 else 'orange' if iou >= 75 else 'red' for iou in mean_ious]
    
    plt.figure(figsize=(12, 8))
    plt.bar(classes, mean_ious, color=colors, alpha=0.7)
    plt.title('Performance IoU par Classe', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Mean IoU (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='Excellent (90%)')
    plt.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='Bon (75%)')
    plt.legend()
    plt.tight_layout()
    
    chart_path = img_dir / "performance_by_class.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"img/{chart_path.name}"

def create_stability_chart(json_data, img_dir):
    """Crée un scatter plot IoU vs Stabilité"""
    class_stability = json_data['stability_metrics']['class_stability']
    
    classes = list(class_stability.keys())
    mean_ious = [stats['mean_iou'] * 100 for stats in class_stability.values()]
    std_ious = [stats['std_iou'] * 100 for stats in class_stability.values()]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(mean_ious, std_ious, alpha=0.7, s=100)
    
    for i, class_name in enumerate(classes):
        plt.annotate(class_name, (mean_ious[i], std_ious[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.title('Performance vs Stabilité', fontsize=16, fontweight='bold')
    plt.xlabel('Mean IoU (%)', fontsize=12)
    plt.ylabel('Écart-type (%)', fontsize=12)
    plt.axvline(x=90, color='green', linestyle='--', alpha=0.5)
    plt.axvline(x=75, color='orange', linestyle='--', alpha=0.5)
    plt.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Seuil instabilité')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    chart_path = img_dir / "performance_vs_stability.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"img/{chart_path.name}"

def create_frequency_chart(json_data, img_dir):
    """Crée un bar chart de la fréquence d'apparition des classes"""
    class_frequency = json_data['class_frequency']
    total_images = json_data['global_metrics']['total_images']
    
    # Exclure Background pour plus de lisibilité
    filtered_frequency = {k: v for k, v in class_frequency.items() if k != 'Background'}
    
    classes = list(filtered_frequency.keys())
    frequencies = [freq/total_images*100 for freq in filtered_frequency.values()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, frequencies, color='skyblue', alpha=0.7)
    plt.title('Fréquence d\'Apparition des Classes', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Fréquence (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Ajouter les valeurs sur les barres
    for bar, freq in zip(bars, frequencies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{freq:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    chart_path = img_dir / "class_frequency.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return f"img/{chart_path.name}"

def create_class_table(classes_list, title):
    """Crée un tableau Markdown pour les classes"""
    if not classes_list:
        return "*Aucune classe dans cette catégorie*\n"
    
    table = "| Classe | Mean IoU | Écart-type | Stabilité |\n"
    table += "|--------|----------|------------|----------|\n"
    
    for class_name, mean_iou, std_iou in classes_list:
        stability = "🟢 Stable" if std_iou < 0.1 else "🟡 Variable" if std_iou < 0.2 else "🔴 Instable"
        table += f"| {class_name} | {mean_iou*100:.1f}% | ±{std_iou*100:.1f}% | {stability} |\n"
    
    return table

def fill_template_and_save(json_data, template_path="templates/template_report.md", output_dir="reports"):
    """Remplit le template et sauvegarde le rapport final"""
    
    # Créer les dossiers de sortie
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "img").mkdir(exist_ok=True)
    
    # Lire le template
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    # Préparer les données
    global_metrics = json_data['global_metrics']
    class_stability = json_data['stability_metrics']['class_stability']
    
    # Formatage des pourcentages
    mean_iou_percent = f"{global_metrics['mean_iou'] * 100:.1f}"
    pixel_accuracy_percent = f"{global_metrics['pixel_accuracy']:.1f}"
    std_iou_percent = f"{json_data['stability_metrics']['std_iou'] * 100:.1f}"
    
    # Classification des classes
    excellent_classes = []
    good_classes = []
    problematic_classes = []
    
    for class_name, stats in class_stability.items():
        mean_iou = stats['mean_iou']
        if mean_iou >= 0.9:
            excellent_classes.append((class_name, mean_iou, stats['std_iou']))
        elif mean_iou >= 0.75:
            good_classes.append((class_name, mean_iou, stats['std_iou']))
        else:
            problematic_classes.append((class_name, mean_iou, stats['std_iou']))
    
    # Générer les charts
    charts_paths = generate_charts(json_data, output_dir / "img")
    
    # Copier les images de résultats
    result_images, worst_number = copy_result_images(json_data['performance_ranking'], output_dir)
    # Nom de la pire image pour récuperer ses datas dans le json
    worst_mask = f"mask_{worst_number}.png"
    worst_mask_data = next((img for img in json_data['per_image_results'] if img['image'] == worst_mask), None)    # Générer les tableaux et analyses
    # Formatage élégant des données de la pire image
    worst_mask_data_formatted = format_worst_image_analysis(worst_mask_data)
    excellent_table = create_class_table(excellent_classes, "Excellentes")
    good_table = create_class_table(good_classes, "Bonnes")
    problematic_table = create_class_table(problematic_classes, "Problématiques")
    
    stability_analysis = generate_stability_analysis(json_data)
    warning_classes = generate_warning_analysis(json_data)
    best_images_table = create_images_table(json_data['performance_ranking']['best_5'], "Meilleures")
    worst_images_table = create_images_table(json_data['performance_ranking']['worst_5'], "Pires")
    
    # Remplacer les placeholders
    filled_template = template.format(
        total_images=global_metrics['total_images'],
        mean_iou_percent=mean_iou_percent,
        pixel_accuracy_percent=pixel_accuracy_percent,
        std_iou_percent=std_iou_percent,
        excellent_classes_table=excellent_table,
        good_classes_table=good_table,
        problematic_classes_table=problematic_table,
        stability_analysis=stability_analysis,
        warning_classes=warning_classes,
        best_images_table=best_images_table,
        worst_images_table=worst_images_table,
        best_image_visual=result_images.get('best', '*Image non disponible*'),
        titre_best_image=result_images.get('titre_best_image', ''),
        worst_image_visual=result_images.get('worst', '*Image non disponible*'),
        titre_worst_image=result_images.get('titre_worst_image', ''),
        worst_mask_data=worst_mask_data_formatted if worst_mask_data_formatted else "*Données non disponibles*",
        performance_chart=charts_paths['performance'],
        stability_chart=charts_paths['stability'],
        frequency_chart=charts_paths['frequency']
    )
    
    # Sauvegarder le rapport
    output_file = output_dir / "evaluation_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(filled_template)
    
    print(f"Rapport généré : {output_file}")
    print(f"Images copiées : {len(result_images)} images de résultats")
    return output_file