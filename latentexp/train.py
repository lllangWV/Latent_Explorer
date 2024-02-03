import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from latentexp.model import MyModel


# Hyperparameters
num_epochs = 10
lr=0.001
batch_size = 32

# Set up your training data
train_dataset = ...
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Set up your model, loss function, and optimizer
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print training progress
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
