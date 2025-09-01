CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17
}

LABELS_MAPPING = {
    "0": "Background",
    "1": "Hat",
    "2": "Hair",
    "3": "Sunglasses",
    "4": "Upper-clothes",
    "5": "Skirt",
    "6": "Pants",
    "7": "Dress",
    "8": "Belt",
    "9": "Left-shoe",
    "10": "Right-shoe",
    "11": "Face",
    "12": "Left-leg",
    "13": "Right-leg",
    "14": "Left-arm",
    "15": "Right-arm",
    "16": "Bag",
    "17": "Scarf"
}

COLOR_MAPPING = {
    1: (255, 255, 0),   # JAUNE - Hat
    2: (255, 165, 0),   # ORANGE - Hair  
    3: (255, 0, 255),   # MAGENTA - Sunglasses
    4: (255, 0, 0),     # ROUGE - Upper-clothes
    5: (0, 255, 255),   # CYAN - Skirt
    6: (0, 255, 0),     # VERT - Pants
    7: (0, 0, 255),     # BLEU - Dress
    8: (128, 0, 128),   # VIOLET - Belt
    9: (255, 140, 0),   # ORANGE FONCÉ - Left-shoe
    10: (139, 69, 19),  # MARRON - Right-shoe
    11: (255, 220, 177), # BEIGE CLAIR - Face
    12: (205, 170, 125), # BEIGE MOYEN - Left-leg
    13: (185, 150, 105), # BEIGE FONCÉ - Right-leg
    14: (225, 190, 145), # BEIGE ROSÉ - Left-arm
    15: (165, 130, 85),  # BEIGE OLIVE - Right-arm
    16: (255, 82, 243),  # VIOLET CLAIR - Bag
    17: (255, 20, 147)  # ROSE - Scarf
}


API_SEGMENTATION_OUTPUTS_DIR = "content/top_influenceurs_2024/Output_API"
IMG_DIR = "content/top_influenceurs_2024/IMG"
MASK_DIR = "content/top_influenceurs_2024/Mask"
EXPECTED_SEGMENTATION_OUTPUTS_DIR = "content/top_influenceurs_2024/Expected_Results"
WWG_SEGMENTATION_OUTPUTS_DIR = "content/top_influenceurs_2024/Real_Results"
LOG_DIR = "logs"