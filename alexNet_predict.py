import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# 1. DEFINE THE MODEL ARCHITECTURE (Must match your trained model)
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

class AlexNetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AlexNet Image Classifier")
        self.root.geometry("500x600")

        # Load Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.model = AlexNet8Layers(num_classes=10).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load('direct_alexnet_90epochs.pth', map_location=self.device))
            self.model.eval()
        except:
            messagebox.showerror("Error", "Model file 'direct_alexnet_90epochs.pth' not found!")

        # UI Elements
        self.label_title = tk.Label(root, text="AlexNet Prediction (CIFAR-10)", font=("Helvetica", 16, "bold"))
        self.label_title.pack(pady=20)

        self.btn_browse = tk.Button(root, text="Select Image", command=self.upload_image, font=("Helvetica", 12), bg="#4CAF50", fg="white", padx=20)
        self.btn_browse.pack(pady=10)

        self.canvas = tk.Label(root) # To display the image
        self.canvas.pack(pady=20)

        self.label_result = tk.Label(root, text="Result: None", font=("Helvetica", 14, "bold"), fg="blue")
        self.label_result.pack(pady=10)

        self.label_conf = tk.Label(root, text="Confidence: 0%", font=("Helvetica", 12))
        self.label_conf.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        # Display Image in GUI
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img_display = ImageTk.PhotoImage(img)
        self.canvas.configure(image=img_display)
        self.canvas.image = img_display

        # Perform Prediction
        self.predict(file_path)

    def predict(self, path):
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        img = Image.open(path).convert('RGB')
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0).to(self.device)

        with torch.no_grad():
            output = self.model(batch_t)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

        res_text = f"Result: {self.classes[predicted.item()].upper()}"
        conf_text = f"Confidence: {confidence.item() * 100:.2f}%"

        self.label_result.config(text=res_text)
        self.label_conf.config(text=conf_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = AlexNetGUI(root)
    root.mainloop()