import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from google.colab import files
import io
import matplotlib.pyplot as plt

# 1. MODEL ARCHITECTURE (Must match your 8-layers)
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

# 2. LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet8Layers(num_classes=10).to(device)
try:
    model.load_state_dict(torch.load('direct_alexnet_90epochs.pth', map_location=device))
    model.eval()
    print("✅ Model loaded successfully!")
except:
    print("❌ Error: 'direct_alexnet_90epochs.pth' not found in Colab files.")

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 3. UPLOAD AND PREDICT FUNCTION
print("\nClick the button below to upload an image:")
uploaded = files.upload()

for filename in uploaded.keys():
    # Load and Pre-process Image
    img_data = uploaded[filename]
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img_t = transform(img).unsqueeze(0).to(device)

    # Perform Prediction
    with torch.no_grad():
        output = model(img_t)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    # Show Result
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {classes[predicted.item()].upper()}\nConfidence: {confidence.item() * 100:.2f}%")
    plt.show()