import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Path to the CSV file
current_dir = os.getcwd()
file_path = os.path.join(current_dir,'data', 'training_data.csv')
model_save_path = os.path.join(current_dir,'model', 'neural_net_classifier.pth')

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(file_path, sep=' ', names=["F1", "F2", "F3", "F4", "label"])



label_encoder = LabelEncoder()
data_copy = data.copy()
labels = np.array(data_copy.pop('label'))
encoded_labels = label_encoder.fit_transform(labels)
data_copy['label'] = encoded_labels



# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.data = torch.tensor(dataframe, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = MyDataset(np.array(data_copy))



# Specify the split ratio
train_ratio = 0.8  # 80% for training, 20% for testing

# Calculate the number of samples for train and test sets
train_size = int(dataset.data.shape[0] * train_ratio)
test_size = dataset.data.shape[0] - train_size


# Use torch.utils.data.random_split to split the dataset
train_dataset, test_dataset = torch.utils.data.random_split(dataset.data, [train_size, test_size])

# Create data loaders for train and test sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


# Define the neural network classifier
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the hyperparameters
input_size = dataset.data.shape[1] - 1  # Number of input features
hidden_size = 64
num_classes = 2  # Number of output classes


# Create an instance of the classifier
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

classifier = Classifier(input_size, hidden_size, num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.01)

num_epochs = 1
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs = batch[:, :-1].to(device) # Extract input features
        labels = batch[:, -1].to(device).to(torch.int64) # Extract labels

        optimizer.zero_grad()

        # Forward pass
        outputs = classifier(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward and optimize
        
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")


torch.save(classifier.state_dict(), model_save_path)


#Syntax for loading model
#classifier = Classifier(input_size, hidden_size, num_classes).to(device)
#classifier.load_state_dict(torch.load(model_save_path))


correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        inputs = batch[:, :-1].to(device) # Extract input features
        labels = batch[:, -1].to(device).to(torch.int64) # Extract labels



        # calculate outputs by running images through the network
        outputs = classifier(inputs)

        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print(f'Accuracy of the network on the test set is: {100 * correct // total} %')

