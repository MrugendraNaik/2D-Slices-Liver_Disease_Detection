import os
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/liver_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

class_names = ["healthy", "disease"]
test_dir = "data/test"

for filename in os.listdir(test_dir):
    if not filename.endswith(".jpg"):
        continue

    img_path = os.path.join(test_dir, filename)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        _, predicted = torch.max(probs, 1)

    print(f"Image: {filename}")
    print(f"Predicted: {class_names[predicted.item()]}")
    print(f"Confidence: {probs.squeeze().tolist()}")

    plt.imshow(img)
    plt.title(f"{class_names[predicted.item()]} ({probs.squeeze()[predicted.item()]:.2f})")
    plt.axis("off")
    plt.show()

