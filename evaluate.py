# evaluate.py
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from io import BytesIO
import json
from inference import DRModel  # Your model class

# -----------------------------
# Load dataset
# -----------------------------
data_dir = "D:/datasets/dataset_eyepacs/EyePACSDataset"
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
dfs = [pd.read_parquet(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# Simple 80/20 split
split_idx = int(0.8 * len(df))
test_df = df[split_idx:].reset_index(drop=True)

# -----------------------------
# Create Dataset & DataLoader
# -----------------------------
class EyePACSTestDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        label = row["label_code"]
        if self.transform:
            img = self.transform(img)
        return img, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = EyePACSTestDataset(test_df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -----------------------------
# Load trained model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DRModel("models_weights/dr_model.pth")
model.model.to(device)
model.model.eval()

# -----------------------------
# Evaluate
# -----------------------------
all_labels = []
all_preds = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model.model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

accuracy = 100 * correct / total
print(f"\n✅ Overall Test Accuracy: {accuracy:.2f}%\n")

# -----------------------------
# Confusion Matrix & Metrics
# -----------------------------
num_classes = 5
class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

cm = torch.zeros(num_classes, num_classes, dtype=torch.int32)
for t, p in zip(all_labels, all_preds):
    cm[t, p] += 1

print("Confusion Matrix (rows=true, cols=predicted):")
print(cm)

metrics = {
    "accuracy": round(accuracy, 2),
    "per_class": {},
    "confusion_matrix": cm.tolist()
}

print("\n📊 Per-Class Metrics:")
f1_scores = []
support = []  # number of samples per class

for i in range(num_classes):
    TP = cm[i, i].item()
    FP = cm[:, i].sum().item() - TP
    FN = cm[i, :].sum().item() - TP

    # Formula definitions
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    f1_scores.append(f1)
    support.append(cm[i, :].sum().item())

    metrics["per_class"][class_names[i]] = {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "support": support[-1]
    }

    # Print with formulas
    print(f"\n📌 {class_names[i]}:")
    print(f"   TP={TP}, FP={FP}, FN={FN}")
    print(f"   Precision = TP / (TP + FP) = {precision*100:.2f}%")
    print(f"   Recall    = TP / (TP + FN) = {recall*100:.2f}%")
    print(f"   F1 Score  = 2 × (Precision × Recall) / (Precision + Recall) = {f1*100:.2f}%")

# -----------------------------
# Overall F1 Scores
# -----------------------------
macro_f1 = sum(f1_scores) / num_classes
micro_f1 = correct / total  # same as accuracy in multi-class
weighted_f1 = sum(f * s for f, s in zip(f1_scores, support)) / sum(support)

metrics["macro_f1"] = round(macro_f1 * 100, 2)
metrics["micro_f1"] = round(micro_f1 * 100, 2)
metrics["weighted_f1"] = round(weighted_f1 * 100, 2)
metrics["accuracy"] = round(accuracy, 2)

print(f"\n📊 Overall Metrics:")
print(f"   Macro F1    = {macro_f1*100:.2f}%")
print(f"   Micro F1    = {micro_f1*100:.2f}% ")
print(f"   Weighted F1 = {weighted_f1*100:.2f}%")
print(f"   Accuracy    = {accuracy:.2f}%")

# -----------------------------
# Save metrics
# -----------------------------
os.makedirs("models_weights", exist_ok=True)
with open("models_weights/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n📁 Metrics saved to models_weights/metrics.json")