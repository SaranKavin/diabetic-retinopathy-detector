import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_eyepacs.dataset_eyepacs import EyePACSDataset
from models.cnn_attention import CNNAttention
import torch.nn as nn
import torch.optim as optim
import os

data_dir = r"D:\datasets\dataset_eyepacs\EyePACSDataset"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = EyePACSDataset(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNAttention(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

os.makedirs("models_weights", exist_ok=True)
torch.save(model.state_dict(), "models_weights/dr_model.pth")
print("✅ Model trained and saved at models_weights/dr_model.pth")
