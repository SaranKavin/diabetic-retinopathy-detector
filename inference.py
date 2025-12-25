import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from models.cnn_attention import CNNAttention
from gradcam import GradCAM

# -------------------------------
# Retina Fundus Validation Helper
# -------------------------------
def is_fundus_image(img_path):
    """Check if uploaded image looks like a retina fundus image (circular structure)."""
    img = cv2.imread(img_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    if perimeter == 0:
        return False

    circularity = 4 * np.pi * (area / (perimeter * perimeter))

    # Retina fundus images are fairly circular (0.6–1.0 range works well)
    return circularity > 0.6


# -------------------------------
# Diabetic Retinopathy Model
# -------------------------------
class DRModel:
    def __init__(self, model_path="models_weights/dr_model.pth"):
        # Load model
        self.model = CNNAttention(num_classes=5)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        
        # Transform for input image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # GradCAM for visualization
        self.gradcam = GradCAM(self.model, target_layer=self.model.features[-1])
        
        # DR classes
        self.labels = [
            "No Diabetic Retinopathy",
            "Mild Retinopathy",
            "Moderate Retinopathy",
            "Severe Retinopathy",
            "Proliferative Retinopathy"
        ]

    def predict(self, img_path):
        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)           # Convert logits to probabilities
            confidence, pred_idx = torch.max(probs, 1)     # Get max probability & index

        # Grad-CAM heatmap
        cam = self.gradcam.generate(img_tensor, class_idx=pred_idx.item())
        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        img_np = np.array(img.resize((224, 224)))
        overlay = cv2.addWeighted(img_np, 0.5, cam, 0.5, 0)

        # Save heatmap
        heatmap_path = os.path.join("static", "uploads", "heatmap.png")
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        cv2.imwrite(heatmap_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Return label, heatmap, and confidence percentage
        return self.labels[pred_idx.item()], heatmap_path, confidence.item() * 100
