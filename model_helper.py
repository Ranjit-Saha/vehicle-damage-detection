import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

class_names = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.2):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Cache the model in memory so it doesn't reload on every single click/upload
@st.cache_resource
def load_my_model():
    model = CarClassifierResNet()
    # Using os.path.join prevents Windows single-backslash escape string syntax errors
    model_path = os.path.join("model", "saved_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def predict(uploaded_file):
    # Load image directly out of user memory buffer
    img = Image.open(uploaded_file).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(img).unsqueeze(0)

    # Retrieve the cached model instantly
    trained_model = load_my_model()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]
