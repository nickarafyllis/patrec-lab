import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

#seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

f = 40 # frequency
points_per_sequence = 10  
time = np.linspace(0, 1, points_per_sequence)

num_sequences = 200
# sin and cos sequence generation
sequences = []
for _ in range(num_sequences):
    
    phase = np.random.randn() # add a random pase in each sequence
    sin_points = np.sin(2 * np.pi * f * time + phase)  # sins
    cos_points = np.cos(2 * np.pi * f * time + phase)  # cosines
    sequences.append((sin_points, cos_points))

# transform training data to PyTorch tensors
input_sequences = [torch.FloatTensor(seq[0]).view(-1, 1, 1) for seq in sequences]
target_sequences = [torch.FloatTensor(seq[1]).view(-1, 1, 1) for seq in sequences]
dataset = TensorDataset(torch.stack(input_sequences), torch.stack(target_sequences))

# split dataset to train test
split_ratio = 0.7
num_train = int(len(dataset) * split_ratio)
num_test = len(dataset) - num_train
train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

# Create data loaders for training and testing
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create a sin cos example plots
# fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# # Loop through the values of i (0 and 1)
# for i in range(2):
#     # Extract the data for the selected sequence
#     sin_data, cos_data = sequences[i]

#     # Scatterplot for the sine and cosine waves for the selected sequence
#     ax = axes[i]
#     ax.set_title(f'Scatterplot of Sine and Cosine Waves (i = {i})')
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Value')
#     ax.scatter(range(len(sin_data)), sin_data, label='Sine Wave', marker='o', s=20)
#     ax.scatter(range(len(cos_data)), cos_data, label='Cosine Wave', marker='x', s=20)
#     ax.legend()

# # Show the plots
# plt.tight_layout()
# plt.show()
# plt.close()

# create RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.linear(out)
        return out

input_size = 1
hidden_size = 64
output_size = 1

model = RNN(input_size, hidden_size, output_size)

# we use Mean Square Error loss function and Adam optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# training
num_epochs = 20

for epoch in range(num_epochs):
    total_loss = 0
    for i in range(num_sequences):
        inputs = input_sequences[i]
        targets = target_sequences[i]

        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Mean loss: {total_loss / num_sequences:.4f}')

print('Training Completed.')

# predictions
with torch.no_grad():
    predicted_sequences = [model(seq).view(-1).numpy() for seq in input_sequences[num_train:]]

# Εμφάνιση αποτελεσμάτων
# plt.figure(figsize=(10, 4))
# plt.title('Cosine Prediction')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# for i in range(num_train, num_sequences):
#     plt.scatter(range(len(sequences[i][1])),sequences[i][1], label=f'True cosine points', linestyle='dashed', marker='o', s=20)
#     plt.scatter(range(len(predicted_sequences[i - num_train])), predicted_sequences[i - num_train], label=f'Predicted cosine points', marker='x', s=20)
# plt.legend()
# plt.show()