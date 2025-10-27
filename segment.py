import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Load pretrained U²-Net
def load_model():
    model = torch.hub.load("xuebinqin/U-2-Net", "u2net")
    model.eval()
    return model

def preprocess(image):
    transform = T.Compose([
        T.Resize((320, 320)),
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def segment_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image)

    with torch.no_grad():
        d1, *_ = model(tensor)
        mask = d1.squeeze().cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = (mask > 0.5).astype(np.uint8) * 255

    mask_img = Image.fromarray(mask).resize(image.size)
    overlay = Image.blend(image, Image.merge("RGB", (mask_img, mask_img, mask_img)), alpha=0.3)

    # Save outputs
    mask_img.save("samples/outputs/mask.png")
    overlay.save("samples/outputs/overlay.png")
    return overlay

if __name__ == "__main__":
    model = load_model()
    segment_image(model, "samples/input.jpg")
