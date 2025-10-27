import os, torch
import numpy as np
from PIL import Image
from torchvision import transforms

class FaceSegDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, size=256):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.imgs = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.t = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.size = size

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, self.imgs[i])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.masks[i])).convert("L")
        return self.t(img), torch.from_numpy(np.array(mask)).long()
