import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse
import sys

# 1. DEFINE THE MODEL ARCHITECTURE
# This must match the exact 8-layer structure used during the training phase.
class AlexNet8Layers(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet8Layers, self).__init__()
        
        # FEATURE EXTRACTION: 5 Convolutional Layers
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Layer 2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Layer 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Layer 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Layer 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling to ensure fixed output size (6x6) regardless of input resolution
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # CLASSIFICATION: 3 Fully Connected (Dense) Layers
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
        x = torch.flatten(x, 1) # Flatten the tensor for the classifier
        x = self.classifier(x)
        return x

def predict_now(image_path):
    """
    Loads the trained model weights and performs inference on a single image.
    """
    # Detect hardware (Use NVIDIA GPU if available, else fallback to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CIFAR-10 Dataset Class Names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # IMAGE PRE-PROCESSING
    # Resizing to 224x224 and normalizing using the same parameters used in training
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        # Initialize the model and move it to the selected device
        model = AlexNet8Layers(num_classes=10).to(device)
        
        # Load the saved model weights (.pth file)
        # Using map_location=device ensures compatibility even if loading GPU weights on a CPU
        model.load_state_dict(torch.load('direct_alexnet_90epochs.pth', map_location=device))
        
        # Set the model to evaluation mode (Disables Dropout layers)
        model.eval()

        # Open image and ensure it is in RGB format
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img)
        
        # Add a batch dimension (1, 3, 224, 224) as PyTorch models expect batches
        batch_t = torch.unsqueeze(img_t, 0).to(device)

        # INFERENCE: Perform the forward pass without calculating gradients (saves memory/speed)
        with torch.no_grad():
            output = model(batch_t)
        
        # Calculate probabilities using Softmax
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

        # PRINT FINAL RESULTS
        print(f"\n" + "="*40)
        print(f"       ALEXNET INFERENCE RESULT")
        print(f"="*40)
        print(f" Target File : {image_path}")
        print(f" Prediction  : {classes[predicted.item()].upper()}")
        print(f" Confidence  : {confidence.item() * 100:.2f}%")
        print(f"="*40 + "\n")
        
    except FileNotFoundError:
        print(f"Error: The image file '{image_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Setup Command-Line Argument Parsing
    parser = argparse.ArgumentParser(description="Modern AlexNet Inference Script")
    
    # Define the --image argument (Required)
    parser.add_argument('--image', type=str, required=True, help="Path to the target image for prediction")
    
    args = parser.parse_args()
    
    # Execute the prediction function
    predict_now(args.image)
