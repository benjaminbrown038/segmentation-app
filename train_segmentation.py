import torch, os, numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
import segmentation_models_pytorch as smp
from dataset import FaceSegDataset


os.makedirs("data/images", exist_ok=True)
os.makedirs("data/masks", exist_ok=True)

# Create dummy images if empty
if len(os.listdir("data/images")) == 0:
    from PIL import Image
    import numpy as np
    for i in range(4):
        img = Image.fromarray(np.uint8(np.random.rand(256,256,3)*255))
        mask = Image.fromarray(np.uint8(np.random.randint(0,2,(256,256))))
        img.save(f"data/images/{i}.jpg")
        mask.save(f"data/masks/{i}.png")



def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = FaceSegDataset("data/images", "data/masks")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = smp.DeepLabV3("resnet34", classes=2, encoder_weights="imagenet").to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss={epoch_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "face_seg_model.pth")

if __name__ == "__main__":
    train()
