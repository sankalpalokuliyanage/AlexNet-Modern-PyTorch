import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# 1. Defining the same 8-layer architecture used in training
class AlexNet8Layers(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet8Layers, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def predict_now(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the custom 8-layer model
    model = AlexNet8Layers(num_classes=10).to(device)
    model.load_state_dict(torch.load('direct_alexnet_90epochs.pth'))
    model.eval()

    img = Image.open(image_path).convert('RGB')
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    with torch.no_grad():
        output = model(batch_t)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, predicted = torch.max(probabilities, 0)

    print(f"Prediction: {classes[predicted.item()]}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")

if __name__ == '__main__':
    predict_now('test_image.jpg') # Make sure your dog image is named correctly