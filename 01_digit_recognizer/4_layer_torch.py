# accuracy > 96%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torchvision import transforms

# Load the CSV data
df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

print(len(df))

# Split: first 10,000 rows for testing, rest for training
train_df = df.iloc[10000:]
test_df = df.iloc[:10000]

# Get features and labels
X_train = train_df.drop("label", axis=1).values.reshape(-1, 28, 28).astype("float32")
y_train = train_df["label"].values

X_test = test_df.drop("label", axis=1).values.reshape(-1, 28, 28).astype("float32")
y_test = test_df["label"].values

# Define transformation
transform = transforms.Compose([transforms.ToTensor()])

# Custom Dataset
class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# Datasets and DataLoaders
train_dataset = MNISTDataset(X_train, y_train, transform=transform)
test_dataset = MNISTDataset(X_test, y_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Neural Network
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate model, loss, optimizer
model = DigitRecognizer()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10):
    model.train()
    for X, y in train_loader:
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ✅ Evaluation on the test set
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for X, y in test_loader:
        y_pred = model(X)
        _, predicted = torch.max(y_pred, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
