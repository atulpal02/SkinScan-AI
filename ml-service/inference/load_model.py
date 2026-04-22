import torch
from training.model import get_model

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

NUM_CLASSES = 3

def load_trained_model():
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load("training/checkpoints/model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model