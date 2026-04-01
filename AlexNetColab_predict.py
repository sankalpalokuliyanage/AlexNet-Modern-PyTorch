import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

# 1. Model Architecture (Exact 8-layer AlexNet)
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

# 2. Setup Device and Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet8Layers(num_classes=10).to(device)

# Load the saved model weights
try:
    model.load_state_dict(torch.load('direct_alexnet_90epochs.pth', map_location=device))
    model.eval()
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: 'direct_alexnet_90epochs.pth' not found. Please upload it to Colab.")

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 3. Prediction Function
def predict(inp_img):
    if inp_img is None: return None
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Convert numpy array to PIL Image
    img = Image.fromarray(inp_img.astype('uint8'), 'RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_t)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Return dictionary of top labels
    return {classes[i]: float(probabilities[i]) for i in range(10)}

# 4. Create and Launch Interface
# Changed share=False and added inline=True for Colab stability
interface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(), 
    outputs=gr.Label(num_top_classes=3),
    title="AlexNet CIFAR-10 Classifier",
    description="Upload an image to see the top 3 predictions from your 8-layer AlexNet model."
)

# Use this for Colab to avoid Certificate issues
interface.launch(inline=True, share=False, debug=True)