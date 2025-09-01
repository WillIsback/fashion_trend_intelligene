# 📊 Rapport d'Évaluation du Modèle de Segmentation

## Vue d'ensemble
- **Nombre d'images évaluées** : {total_images}
- **Mean IoU global** : {mean_iou_percent}%
- **Précision des pixels** : {pixel_accuracy_percent}%
- **Stabilité (écart-type)** : ±{std_iou_percent}%

## 🎯 Performance par classe

![Performance par classe]({performance_chart})

### Classes excellentes (IoU ≥ 90%)
{excellent_classes_table}

### Classes bonnes (IoU 75-90%)
{good_classes_table}

### Classes problématiques (IoU < 75%)
{problematic_classes_table}

## 📈 Analyse de stabilité

![Performance vs Stabilité]({stability_chart})

{stability_analysis}

## 📊 Fréquence d'apparition des classes

![Fréquence des classes]({frequency_chart})

## ⚠️ Classes à surveiller
{warning_classes}

## 🏆 Top/Flop images

### Meilleures performances
{best_images_table}

#### Exemple de meilleure segmentation
**{titre_best_image}**
![Meilleure segmentation]({best_image_visual})

### Performances à améliorer
{worst_images_table}

#### Exemple de segmentation problématique
**{titre_worst_image}**

![Segmentation problématique]({worst_image_visual})

{worst_mask_data}
