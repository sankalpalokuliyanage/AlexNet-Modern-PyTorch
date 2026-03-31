import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from torch.cuda.amp import GradScaler, autocast

# 1. Manually defining the 8 Layers of AlexNet
class AlexNet8Layers(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet8Layers, self).__init__()
        
        # --- Feature Extraction (5 Conv Layers) ---
        self.features = nn.Sequential(
            # Layer 1: Conv -> ReLU -> MaxPool
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 2: Conv -> ReLU -> MaxPool
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 3: Conv -> ReLU
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4: Conv -> ReLU
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 5: Conv -> ReLU -> MaxPool
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # --- Classification (3 Fully Connected Layers) ---
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # Layer 6: Linear (FC1)
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # Layer 7: Linear (FC2)
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # Layer 8: Linear (FC3 - Output)
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_direct_alexnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Direct Training on: {torch.cuda.get_device_name(0)}")

    # Data Setup (Using your high RAM capacity)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=16, pin_memory=True)

    # Initialize our custom 8-layer model
    model = AlexNet8Layers(num_classes=10).to(device)
    
    # Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scaler = GradScaler() # For Mixed Precision (Speed)

    epochs = 90
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            with autocast(): # Speed optimization
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
        
        print(f'Epoch [{epoch + 1}/{epochs}] - Average Loss: {running_loss / len(trainloader):.4f}')

    print(f'Total Training Time: {(time.time() - start_time)/60:.2f} mins')
    torch.save(model.state_dict(), 'direct_alexnet_90epochs.pth')

if __name__ == '__main__':
    train_direct_alexnet()