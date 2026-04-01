import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import numpy as np

# 1. MODEL ARCHITECTURE (Exact 8-layer AlexNet)
# This class definition must be identical to the one used during training.
class AlexNet8Layers(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet8Layers, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# 2. SETUP DEVICE AND LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet8Layers(num_classes=10).to(device)

# Load the saved model weights (.pth file)
model_path = 'direct_alexnet_90epochs.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: '{model_path}' not found. Please upload the file to Colab's file sidebar.")

# CIFAR-10 Class Names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 3. PREDICTION FUNCTION
def predict_image(inp_img):
    if inp_img is None:
        return None
    
    # Image Pre-processing (Must match training settings)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Gradio sends image as numpy array, convert to PIL
    img = Image.fromarray(inp_img.astype('uint8'), 'RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_t)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Create a dictionary of results for Gradio Label output
    results = {classes[i]: float(probabilities[i]) for i in range(10)}
    return results

# 4. CREATE GRADIO INTERFACE
interface = gr.Interface(
    fn=predict_image, 
    inputs=gr.Image(), 
    outputs=gr.Label(num_top_classes=3),
    title="AlexNet CIFAR-10 Classifier",
    description="Upload an image to see the top 3 predictions from your custom 8-layer AlexNet model.",
    theme="huggingface"
)

# 5. LAUNCH IN COLAB
# share=False avoids SSL/Certificate issues. 
# inline=True displays the UI directly in the notebook cell.
interface.launch(inline=True, share=False, height=550)