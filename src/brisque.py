import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "brisque_model_live.yml")
range_path = os.path.join(BASE_DIR, "models", "brisque_range_live.yml")

brisque = cv2.quality.QualityBRISQUE_create(model_path, range_path)


def compute_brisque(image):

    score = brisque.compute(image)[0]

    return score