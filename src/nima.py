import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class NIMA(nn.Module):
    def __init__(self, base_model):
        super(NIMA, self).__init__()
        self.base = base_model
        self.dropout = nn.Dropout(0.75)
        self.fc = nn.Linear(2048, 10)  # 10 aesthetic bins
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.base(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.softmax(x)


# Load pretrained ResNet
base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
base_model = nn.Sequential(*list(base_model.children())[:-1])

model = NIMA(base_model)
model.eval()


# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def compute_nima(image):

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        preds = model(img).numpy()[0]

    # Expected score (1–10)
    score = np.sum(preds * np.arange(1, 11))

    return score