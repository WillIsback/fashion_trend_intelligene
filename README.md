# Fashion Trend Intelligence Project

## Overview
The Fashion Trend Intelligence project is designed to perform image segmentation on fashion-related images using the Hugging Face API. The project is structured to facilitate easy maintenance and scalability by organizing the code into separate modules.

## Project Structure
```
fashion_trend_intelligence
├── src
│   ├── api.py
│   ├── utils.py
│   └── processing.py
├── content
│   └── images_a_segmenter
│       └── top_influenceurs_2024
│           └── IMG
├── main.py
├── README.md
├── pyproject.toml
├── huggingface_api_cloth_seg.ipynb
├── output
└── output_segmentation
```

## Modules

### `src/api.py`
This module contains functions and classes related to interacting with the Hugging Face API for image segmentation. The primary function is `call_hf_segmentation_api`, which sends image data to the API and handles the response.

### `src/utils.py`
This module includes utility functions that assist with image processing and mask creation. Key functions include:
- `get_image_dimensions`: Retrieves the dimensions of an image.
- `decode_base64_mask`: Decodes a base64-encoded mask into a NumPy array.
- `create_masks`: Combines multiple class masks into a single segmentation mask.

### `src/processing.py`
This module handles the main processing logic for the images. It includes:
- `segment_images_batch`: Processes a batch of images using the API.
- `save_segmented_images_batch`: Saves the original images and their segmented masks.

### `main.py`
The entry point for the application, which imports functions from the `src` modules to execute the main workflow of loading images, calling the API, and saving results.

## Directories
- `content/images_a_segmenter/top_influenceurs_2024/IMG`: This directory holds the images to be segmented.
- `output` and `output_segmentation`: These directories store the results of the segmentation process.

## Dependencies
The project dependencies and configurations are managed in the `pyproject.toml` file.

## Jupyter Notebook
The `huggingface_api_cloth_seg.ipynb` notebook is provided for experimentation and testing of the segmentation logic.

## Usage
To run the project, execute the `main.py` file. Ensure that the necessary dependencies are installed and that the images are placed in the specified directory.