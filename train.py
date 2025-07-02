import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

# Basic image transforms for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset from the processed images
dataset = datasets.ImageFolder("data/train", transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)  # use 0 for Windows safety

# Load pre-trained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # binary classification: healthy vs disease

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Use weighted loss to handle imbalance
class_counts = [0, 0]
for _, label in dataset.samples:
    class_counts[label] += 1
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights.to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 3
losses = []

print("Starting training...\n")
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 5 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    print(f"✅ Epoch {epoch+1} completed — Average Loss: {epoch_loss:.4f}\n")

# Save the model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/liver_model.pth")
print("🧠 Model saved to models/liver_model.pth")

# Plot loss graph
plt.plot(range(1, num_epochs+1), losses, marker="o")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

