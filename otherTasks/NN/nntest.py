import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
import os
import matplotlib.pyplot as plt  # <-- needed for plotting

train_losses = []
test_accuracies = []

# --- Settings ---
batch_size = 8
epochs = 25
learning_rate = 1e-3
image_size = 64  # Resize all images to 64x64
num_classes = 3  # jab and kick and uppercut

# --- Transforms ---
transform = Compose([
    Resize((image_size, image_size)),
    ToTensor()
])

# --- Dataset & Dataloaders ---
train_data = ImageFolder(root='otherTasks/NN/martial_arts_dataset/train', transform=transform)
test_data = ImageFolder(root='otherTasks/NN/martial_arts_dataset/test', transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# --- Neural Network Model ---
class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(3 * image_size * image_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

model = PoseNet()

# --- Training Setup ---
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Training Loop ---
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0  # <-- added
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item()  # <-- added

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 10 == 0:
            print(f"loss: {loss.item():.4f}")
    train_losses.append(total_loss / len(dataloader))  # <-- added

# --- Testing Loop ---
def test_loop(dataloader, model, loss_fn):
    model.eval()
    total_loss, correct = 0, 0
    size = len(dataloader.dataset)
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
    accuracy = 100 * correct / size  # <-- added
    test_accuracies.append(accuracy)  # <-- added
    print(f"Test Accuracy: {accuracy:.2f}% | Avg Loss: {total_loss / len(dataloader):.4f}")

# --- Run Training ---
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(test_loader, model, loss_fn)

# --- Predict One Sample ---
label_map = train_data.classes  # ['jab', 'kick']
sample, label = test_data[0]
model.eval()
with torch.no_grad():
    prediction = model(sample.unsqueeze(0))
    predicted_label = label_map[prediction.argmax(1).item()]
print(f"\nPredicted: {predicted_label}, Actual: {label_map[label]}")

# --- Plot Loss and Accuracy ---
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label="Test Accuracy", color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy")
plt.grid(True)

plt.tight_layout()
plt.show()
