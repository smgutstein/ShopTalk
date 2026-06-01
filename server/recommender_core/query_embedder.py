
import logging
import torch

from imagebind import data
from imagebind.models.imagebind_model import ModalityType

from langchain_core.embeddings.embeddings import Embeddings


def normalize(vectors):
    norms = torch.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


class QueryEmbedder(Embeddings):
    def __init__(self, ibind_model, device):
        self.ibind_model = ibind_model
        self.device = device

    def embed_documents(self, texts):
        logging.info(f"searching multiple strings: {texts}")
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        logging.info(f"embedding one string: {text}")
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text([text], self.device)
        }
        with torch.no_grad():
            outputs = self.ibind_model(inputs)
            normalized = normalize(outputs[ModalityType.TEXT])
            logging.info(f"text embedding's shape: {outputs[ModalityType.TEXT].shape}")
            logging.info(f"after normalization.  : {normalized.shape}")
        return normalized.cpu().numpy().flatten()

    def embed_image(self, image_path):
        logging.info(f"embedding one image: {image_path}")
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(
                [str(image_path)],
                self.device,
            )
        }
        with torch.no_grad():
            outputs = self.ibind_model(inputs)
            normalized = normalize(outputs[ModalityType.VISION])
            logging.info(f"image embedding's shape: {outputs[ModalityType.VISION].shape}")
            logging.info(f"after normalization.  : {normalized.shape}")
        return normalized.cpu().numpy().flatten()