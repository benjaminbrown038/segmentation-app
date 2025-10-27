# segmentation-app

# 🧩 Semantic Segmentation App

A Streamlit-based semantic segmentation demo using **PyTorch** and **Segmentation Models PyTorch (SMP)**.  
This app predicts pixel-level class labels for uploaded images using a pretrained **DeepLabV3 (ResNet-34)** model.

## 🚀 Features
- Upload any `.jpg`, `.jpeg`, or `.png` image  
- Visualize segmentation overlay (each region colored uniquely)  
- Easily extendable to custom datasets and model fine-tuning  

## 🧩 Tech Stack
- **PyTorch** — deep learning backend  
- **Segmentation Models PyTorch** — pretrained architectures (DeepLab, U-Net, etc.)  
- **Streamlit** — simple browser interface  
- **OpenCV / Pillow / Albumentations** — preprocessing  

## 🛠️ Setup
```bash
git clone https://github.com/<your-username>/semantic-segmentation-app.git
cd semantic-segmentation-app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
