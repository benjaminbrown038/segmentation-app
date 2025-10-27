# segment.py
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

# Load pretrained DeepLabV3 model (for demonstration)
def load_model():
    model = smp.DeepLabV3(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=21,          # matches PASCAL VOC
        activation=None
    )
    model.eval()
    return model

# Preprocess image
def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((256, 256))
    tensor = TF.to_tensor(image).unsqueeze(0)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor

# Perform segmentation and create mask overlay
def segment_image(model, image_path):
    image = Image.open(image_path)
    tensor = preprocess(image)

    with torch.no_grad():
        output = model(tensor)
        mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colors = np.random.randint(0, 255, size=(mask.max() + 1, 3))
    for i in range(mask.max() + 1):
        mask_rgb[mask == i] = colors[i]

    overlay = Image.blend(image.resize(mask_rgb.shape[:2][::-1]), Image.fromarray(mask_rgb), alpha=0.5)
    return overlay, mask
