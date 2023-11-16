import os
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
import torch.nn as nn
import math
from parser import parser
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

output_dim = 10  # number of digits
# TODO: YOUR CODE HERE
# Play with variations of these hyper-parameters and report results
rnn_size = 64
num_layers = 2
bidirectional = True
dropout = 0.4
batch_size = 32
patience = 3
epochs = 15
lr = 1e-3
weight_decay = 0.1

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

class EarlyStopping(object):
    def __init__(self, patience, mode="min", base=None):
        self.best = base
        self.patience = patience
        self.patience_left = patience
        self.mode = mode

    def stop(self, value: float) -> bool:
        # TODO: YOUR CODE HERE
        # Decrease patience if the metric has not improved
        # Stop when patience reaches zero
        if self.mode == "min":
            if self.best is None or value < self.best:
                self.best = value
                self.patience_left = self.patience
            else:
                self.patience_left -= 1
                if self.patience_left == 0:
                    return True  # Stop training
        elif self.mode == "max":
            if self.best is None or value > self.best:
                self.best = value
                self.patience_left = self.patience
            else:
                self.patience_left -= 1
                if self.patience_left == 0:
                    return True  # Stop training
        return False  # Continue training

    def has_improved(self, value: float) -> bool:
        # TODO: YOUR CODE HERE
        # Check if the metric has improved
        if self.mode == "min":
            return value < self.best if self.best is not None else True
        elif self.mode == "max":
            return value > self.best if self.best is not None else True
        return False
    
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def sort_sequences_by_length(sequences, labels):
    sorted_seq_lengths = sorted(enumerate(sequences), key=lambda x: len(x[1]), reverse=True)
    sorted_sequences = [seq for i, seq in sorted_seq_lengths]
    sorted_labels = [labels[i] for i, seq in sorted_seq_lengths]
    return sorted_sequences, sorted_labels

class FrameLevelDataset(Dataset):
    def __init__(self, feats, labels):
        """
        feats: Python list of numpy arrays that contain the sequence features.
               Each element of this list is a numpy array of shape seq_length x feature_dimension
        labels: Python list that contains the label for each sequence (each label must be an integer)
        """
        # TODO: YOUR CODE HERE
        self.lengths = [len(i) for i in feats]
        feats, labels = sort_sequences_by_length(feats, labels)

        self.feats = self.zero_pad_and_stack(feats)
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels).astype("int64")

    def zero_pad_and_stack(self, x: np.ndarray) -> np.ndarray:
        """
        This function performs zero padding on a list of features and forms them into a numpy 3D array
        returns
            padded: a 3D numpy array of shape num_sequences x max_sequence_length x feature_dimension
        """
        max_length = max(len(seq) for seq in x)
        feature_dimension = x[0].shape[1]  # Assuming all sequences have the same feature dimension

        # Initialize padded array with zeros
        num_sequences = len(x)
        padded = np.zeros((num_sequences, max_length, feature_dimension), dtype=np.float32)
        # TODO: YOUR CODE HERE
        # --------------- Insert your code here ---------------- #

        # fill the padded array with sequences, zero-padding where necessary
        for i, seq in enumerate(x):
            seq_len = len(seq)
            padded[i, :seq_len, :] = seq
        
        return padded

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)


class BasicLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        rnn_size,
        output_dim,
        num_layers,
        bidirectional=True,
        dropout=0.0,
        l2_reg=0.0,        
    ):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size

        # TODO: YOUR CODE HERE
        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=rnn_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True
        )

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Output layer
        self.fc = nn.Linear(self.feature_size, output_dim)
        
        # L2 regularization 
        self.l2_reg = l2_reg



    def forward(self, x, lengths):
        """
        x : 3D numpy array of dimension N x L x D
            N: batch index
            L: sequence index
            D: feature index

        lengths: N x 1
        """

        # TODO: YOUR CODE HERE
        # --------------- Insert your code here ---------------- #

        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network
        # Convert numpy array to PyTorch tensor
        #x_tensor = torch.from_numpy(x).float()

        # Pack the sequence based on lengths for handling variable-length sequences
        packed_input = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward pass through LSTM
        packed_output, _ = self.lstm(packed_input)
        # Unpack the output (padded to the original length)
        lstm_out, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)

        # Extract the last output from each sequence
        last_outputs = self.last_timestep(lstm_out, lengths, self.bidirectional)

        # Apply dropout
        last_outputs = self.dropout(last_outputs)

        # Fully connected layer
        last_outputs  = self.fc(last_outputs)

        return last_outputs


    def last_timestep(self, outputs, lengths, bidirectional=True):
        """
        Returns the last output of the LSTM taking into account the zero padding
        """
        # TODO: READ THIS CODE AND UNDERSTAND WHAT IT DOES
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        # TODO: READ THIS CODE AND UNDERSTAND WHAT IT DOES
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # TODO: READ THIS CODE AND UNDERSTAND WHAT IT DOES
        # Index of the last output for each sequence.
        idx = (
            (lengths - 1)
            .view(-1, 1)
            .expand(outputs.size(0), outputs.size(2))
            .unsqueeze(1)
        )
        return outputs.gather(1, idx).squeeze()


def create_dataloaders(batch_size):
    X, X_test, y, y_test, spk, spk_test = parser("recordings", n_mfcc=13)

    X_train, X_val, y_train, y_val, spk_train, spk_val = train_test_split(
        X, y, spk, test_size=0.2, stratify=y
    )

    trainset = FrameLevelDataset(X_train, y_train)
    validset = FrameLevelDataset(X_val, y_val)
    testset = FrameLevelDataset(X_test, y_test)
    # TODO: YOUR CODE HERE
    # Initialize the training, val and test dataloaders (torch.utils.data.DataLoader)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(validset, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


def training_loop(model, train_dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    num_batches = 0
    for num_batch, batch in enumerate(train_dataloader):
        features, labels, lengths = batch
        
        # TODO: YOUR CODE HERE
        optimizer.zero_grad() # zero grads in the optimizer
        outputs = model(features, lengths)# run forward pass
        loss = criterion(outputs, labels)# calculate loss
        
        # TODO: YOUR CODE HERE
        loss.backward()  # Run backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()
        num_batches += 1
    train_loss = running_loss / num_batches
    return train_loss


def evaluation_loop(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    y_pred = torch.empty(0, dtype=torch.int64)
    y_true = torch.empty(0, dtype=torch.int64)
    with torch.no_grad():
        for num_batch, batch in enumerate(dataloader):
            features, labels, lengths = batch

            # TODO: YOUR CODE HERE
            # Run forward pass
            logits = model(features, lengths)
            # calculate loss
            loss = criterion(logits, labels)
            running_loss += loss.item()
            # Predict
            _, outputs = torch.max(logits, 1)  # Calculate the argmax of logits
            y_pred = torch.cat((y_pred, outputs))
            y_true = torch.cat((y_true, labels))
            num_batches += 1
    valid_loss = running_loss / num_batches
    return valid_loss, y_pred, y_true


def train(train_dataloader, val_dataloader, criterion):
    # TODO: YOUR CODE HERE
    input_dim = train_dataloader.dataset.feats.shape[-1]
    model = BasicLSTM(
        input_dim,
        rnn_size,
        output_dim,
        num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
    )
    # TODO: YOUR CODE HERE
    # Initialize AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    
    early_stopping = EarlyStopping(patience, mode="min")
    for epoch in range(epochs):
        training_loss = training_loop(model, train_dataloader, optimizer, criterion)
        valid_loss, y_pred, y_true = evaluation_loop(model, val_dataloader, criterion)

        train_losses.append(training_loss)
        val_losses.append(valid_loss)
        
        # TODO: Calculate and print accuracy score
        correct = (y_pred == y_true).sum().item()
        valid_accuracy = correct / len(y_pred)
        print(
            "Epoch {}: train loss = {}, valid loss = {}, valid acc = {}".format(
                epoch, training_loss, valid_loss, valid_accuracy
            )
        )
        if early_stopping.stop(valid_loss):
            print("early stopping...")
            break
        
        # If validation loss has decreased, save the model
        if early_stopping.has_improved(valid_loss):
            early_stopping.save_checkpoint(valid_loss, model)
       
    # plot train-validation losses    
    plot_loss(train_losses, val_losses)
    
    # Load the best model weights from the checkpoint
    #model.load_state_dict(torch.load('checkpoint.pt'))

    return model


train_dataloader, val_dataloader, test_dataloader = create_dataloaders(batch_size)
# TODO: YOUR CODE HERE
# Choose an appropriate loss function
criterion = torch.nn.CrossEntropyLoss() #because it is multilabel classification

model = train(train_dataloader, val_dataloader, criterion)

test_loss, test_pred, test_true = evaluation_loop(model, test_dataloader, criterion)


# TODO: YOUR CODE HERE
# print test loss and test accuracy
correct_test = (test_pred == test_true).sum().item()
total_test = len(test_pred)
test_accuracy = correct_test / total_test if total_test > 0 else 0.0

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Convert tensors to numpy arrays
test_true_np = test_true.numpy()
test_pred_np = test_pred.numpy()

# Calculate the confusion matrix
conf_matrix = confusion_matrix(test_true_np, test_pred_np)

print("Confusion Matrix:")
print(conf_matrix)

# Displaying the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', 
            xticklabels=np.unique(test_true_np), yticklabels=np.unique(test_true_np))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
