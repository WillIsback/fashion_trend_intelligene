from .utils import get_logger
import os
import requests

logger = get_logger(__name__, __name__ + ".log")

def call_hf_segmentation_api(image_data, model="sayeed99/segformer_b3_clothes"):
    """
    Calls the Hugging Face segmentation API with the provided image data.

    Args:
        image_data (bytes): The image data to be sent to the API.
        model (str): The model to be used for segmentation.

    Returns:
        dict: The response from the API containing segmentation results.
    """
    API_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
    }

    try:
        with open(image_data, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers={"Content-Type": "image/jpeg", **headers}, data=data)
        logger.info(f"API Response Status Code: {response.status_code}")
        logger.info(f"API Response Content: {response.content}")
        if response.status_code == 200:
            return response.json()
        else:
            logger.info(f"Error calling Hugging Face API: {response.status_code} - {response.text}")
            print(f"Erreur API Hugging Face ({response.status_code}): {response.json().get('error', response.text)}")
            return None

    except Exception as e:
        logger.info(f"Exception during API call: {e}")
        print(f"Exception lors de l'appel à l'API Hugging Face: {e}")
        return None