import torch
import numpy as np
import cv2
from torchvision import transforms
from .load_model import load_trained_model, DEVICE
from explainability.gradcam import GradCAM, overlay_heatmap

# load once (important)
model = load_trained_model()
gradcam = GradCAM(model, model.layer4[-1])

# class labels (VERY IMPORTANT — must match folder names)
CLASS_NAMES = ["acne", "melanoma", "nevus"]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(image):
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    # 🔥 Grad-CAM (needs gradients → no torch.no_grad here)
    cam = gradcam.generate(input_tensor, predicted.item())

    heatmap = overlay_heatmap(image, cam)
    

   
    
    return {
        "prediction": CLASS_NAMES[predicted.item()],
        "confidence": round(float(confidence.item()), 3),
        "heatmap": heatmap
        
    }