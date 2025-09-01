import base64
import io
from PIL import Image
import numpy as np
import logging
import cv2
import os
from .config import CLASS_MAPPING, LABELS_MAPPING, COLOR_MAPPING, LOG_DIR

def get_image_dimensions(img_path):
    """
    Get the dimensions of an image.

    Args:
        img_path (str): Path to the image.

    Returns:
        tuple: (width, height) of the image.
    """
    original_image = Image.open(img_path)
    return original_image.size

def decode_base64_mask(base64_string, width, height):
    """
    Decode a base64-encoded mask into a NumPy array.

    Args:
        base64_string (str): Base64-encoded mask.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Single-channel mask array.
    """
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)


def create_masks(results, width, height):
    """
    Combine multiple class masks into a single segmentation mask.

    Args:
        results (list): List of dictionaries with 'label' and 'mask' keys.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: Combined segmentation mask with class indices.
    """
    combined_mask = np.zeros((height, width), dtype=np.uint8)  # Initialize with Background (0)

    # Process non-Background masks first
    for result in results:
        label = result['label']
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:  # Skip Background
            continue
        mask_array = decode_base64_mask(result['mask'], width, height)
        combined_mask[mask_array > 0] = class_id

    # Process Background last to ensure it doesn't overwrite other classes unnecessarily
    # (Though the model usually provides non-overlapping masks for distinct classes other than background)
    for result in results:
        if result['label'] == 'Background':
            mask_array = decode_base64_mask(result['mask'], width, height)
            combined_mask[mask_array > 0] = 0 # Class ID for Background is 0

    return combined_mask

def get_logger(name=__name__, log_file='app.log'):
    """Retourne un logger configuré"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Handler pour fichier
        file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_file), mode='a')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Handler pour console (optionnel)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        # logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    
    return logger


def colorize_mask(mask, colormap):
    """
    Applique le colormap personnalisé au masque.
    Pour chaque pixel, s'il correspond à un label défini dans colormap,
    la couleur correspondante est assignée.
    """
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in colormap.items():
        colored_mask[mask == label] = color
    return colored_mask


def add_legend(image, legend, start_x=10, start_y=10, box_size=15, spacing=5):
    """
    Ajoute une légende sur l'image.
    Pour chaque label, dessine un rectangle de la couleur correspondante et le texte associé.
    """
    img_with_legend = image.copy()
    y = start_y
    for label, text in legend.items():
        # Récupération de la couleur du label
        color = COLOR_MAPPING.get(int(label), (255, 255, 255))
        # Dessin d'un petit rectangle rempli
        cv2.rectangle(img_with_legend, (start_x, y), (start_x + box_size, y + box_size), color, -1)
        # Ajout du texte à droite du rectangle
        cv2.putText(img_with_legend, text, (start_x + box_size + spacing, y + box_size - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += box_size + spacing
    return img_with_legend


# Charger les images et les masques depuis un répertoire local
def load_local_dataset(image_dir, mask_dir):
    
    #Verifier si les répertoires existent
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Le répertoire des images '{image_dir}' n'existe pas.")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Le répertoire des masques '{mask_dir}' n'existe pas.")

    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    paires = []

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        img_filename = os.path.basename(img_path)
        idx = int(''.join(filter(str.isdigit, img_filename)))  # Extraire les chiffres du nom de fichier
        
        image = cv2.imread(img_path) # chargement de l'image originale en couleur
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # chargement du masque en niveaux de gris

        if image is not None and mask is not None:
            paires.append((image, mask, idx))
        else:
            print(f"Warning: Could not read {img_path} or {mask_path}")

    return paires



def save_results(image_dir, mask_dir, output_dir):
    paires = load_local_dataset(image_dir, mask_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img, msk, idx in paires:
        # Colorisation du masque avec le colormap personnalisé
        result_filename = f"result_{idx}.png"
        
        colored_mask = colorize_mask(msk, COLOR_MAPPING)

        # Ajout de la légende sur le masque colorisé
        colored_mask_with_legend = add_legend(colored_mask, LABELS_MAPPING)

        # Superposition du masque coloré sur l'image originale
        overlay = cv2.addWeighted(img, 0.7, colored_mask, 0.3, 0)
        overlay_with_legend = add_legend(overlay, LABELS_MAPPING)

        # Concatenation des images sur une seule ligne
        concatenated = np.hstack([img, colored_mask_with_legend, overlay_with_legend])

        # Sauvegarde de l'image résultante
        output_path = os.path.join(output_dir, result_filename)
        cv2.imwrite(output_path, concatenated)