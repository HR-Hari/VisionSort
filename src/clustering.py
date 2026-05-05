import torch
import numpy as np
from sklearn.cluster import DBSCAN

from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms


# Load model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def extract_features(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model(img)

    feat = features.squeeze().numpy()

    # Normalize (CRITICAL)
    feat = feat / np.linalg.norm(feat)

    return feat


def cluster_images(images):

    feature_list = []

    for name, img in images:
        feature_list.append(extract_features(img))

    feature_array = np.array(feature_list)

    clustering = DBSCAN(
        eps=0.1,              # tuned for cosine
        min_samples=2,
        metric='cosine'
    ).fit(feature_array)

    labels = clustering.labels_

    clusters = {}

    for i, label in enumerate(labels):

        # Handle noise as its own cluster
        if label == -1:
            label = f"noise_{i}"

        if label not in clusters:
            clusters[label] = []

        clusters[label].append(images[i])

    return clusters

def cluster_images_from_features(features, paths):

    clustering = DBSCAN(
        eps=0.3,
        min_samples=2,
        metric='cosine'
    ).fit(features)

    labels = clustering.labels_

    clusters = {}

    for i, label in enumerate(labels):

        if label == -1:
            label = f"noise_{i}"

        if label not in clusters:
            clusters[label] = []

        clusters[label].append(paths[i])

    return clusters