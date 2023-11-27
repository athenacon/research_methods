 
# 2nd implementation with a supervised approach
# I give normal and attacked data in the autoencoder and are classfidied as 0 and 1 (normal or attacked) 
import numpy as np
import torch.nn as nn
import torch
import torchvision
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, roc_curve, auc,  precision_recall_curve
 

class AE_SUPERVISED(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.encoder_hidden_layer1 = nn.Linear(in_features=200, out_features=128)
        self.encoder_hidden_layer2 = nn.Linear(in_features=128, out_features=64)
        self.encoder_hidden_layer3 = nn.Linear(in_features=64, out_features=32)
        self.classifier = nn.Linear(in_features=32, out_features=2) 

    def forward(self, features):
        activation = self.encoder_hidden_layer1(features)
        activation = torch.relu(activation)

        activation = self.encoder_hidden_layer2(activation)
        activation = torch.relu(activation)

        activation = self.encoder_hidden_layer3(activation)
        code = torch.relu(activation)

        clf_output = self.classifier(code)  # No softmax here.
        return clf_output  # Returns raw scores. Loss function handles inherently softmax operation.

def create_x_y_data(grouped_data):
    x_data = []
    labels = []

    for group in grouped_data:
        # Split the data into 'x' and 'y' based on the 'Vehicle' column
        x_temp = group[group['Vehicle'] == 'x'][cols]

        # Add labels
        label = group['Label'].iloc[0]  # Take the 'Attack' label of the group

        labels.append(torch.tensor(label).float())
        if not x_temp.empty and len(group) == 25:
            x_data.append(torch.from_numpy(x_temp.values.flatten()).float())

    # Stack all tensors along a new dimension
    x_data = torch.stack(x_data, dim=0)
    labels = torch.stack(labels, dim=0)

    return x_data, labels

def test_model(model_at_the_best_epoch, dataloader, device):
    CM = np.zeros((2, 2))

    model_at_the_best_epoch.to(device)
    model_at_the_best_epoch.eval()
    accuracies = []
    with torch.no_grad():
        for batch, labels  in dataloader:
            batch = batch.view(-1, 200).to(device)
            labels = labels.long().to(device)
            outputs = model_at_the_best_epoch(batch)
            _, preds = torch.max(outputs, 1)
            CM += confusion_matrix(labels.cpu(), preds.cpu(), labels=[0,1])
            batch_acc = (torch.sum(preds == labels).item()) / labels.size(0)
            accuracies.append(batch_acc)
        tn, fp, fn, tp = CM.ravel()
        acc = np.sum(np.diag(CM) / np.sum(CM))
        sensitivity = tp / (tp + fn)
        precision = tp / (tp + fp)

        print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
        print()
        print('Confusion Matrix : ')
        print(CM)
        print('- Sensitivity : ', (tp / (tp + fn)) * 100)
        print('- Specificity : ', (tn / (tn + fp)) * 100)
        print('- Precision: ', (tp / (tp + fp)) * 100)
        print('- NPV: ', (tn / (tn + fn)) * 100)
        print('- F1 : ', ((2 * sensitivity * precision) / (sensitivity + precision)) * 100)
        print()

    return acc, CM, accuracies
def test_model_pr_roc(model_at_the_best_epoch, dataloader, device):
    CM = np.zeros((2, 2))
    
    model_at_the_best_epoch.to(device)
    model_at_the_best_epoch.eval()
    
    accuracies = []
    all_probs = [] # Store all probabilities here
    all_labels = [] # And all true labels here
    all_preds = []
    with torch.no_grad():
        for batch, labels in dataloader:
            batch = batch.view(-1, 200).to(device)
            labels = labels.long().to(device)
            
            outputs = model_at_the_best_epoch(batch)
            
            # Apply softmax since output returns raw scores
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            # Store probabilities and true labels
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            _, preds = torch.max(outputs, 1)
            # chi square
            all_preds.extend(preds.cpu().numpy())
            
            CM += confusion_matrix(labels.cpu(), preds.cpu(), labels=[0,1])
            batch_acc = (torch.sum(preds == labels).item()) / labels.size(0)
            accuracies.append(batch_acc)

    tn, fp, fn, tp = CM.ravel()
    acc = np.sum(np.diag(CM) / np.sum(CM))
    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp)

    print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
    print('Confusion Matrix : ')
    print(CM)
    print('- Sensitivity : ', (tp / (tp + fn)) * 100)
    print('- Specificity : ', (tn / (tn + fp)) * 100)
    print('- Precision: ', (tp / (tp + fp)) * 100)
    print('- NPV: ', (tn / (tn + fn)) * 100)
    print('- F1 : ', ((2 * sensitivity * precision) / (sensitivity + precision)) * 100)
    
    # Calculate ROC AUC and return values for plotting
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_probs)
    # chi square 
    np.save('data/graph/all_preds_model_2.npy', all_preds)       # Predictions
    np.save('data/graph/all_labels_model_2.npy', all_labels)     # True labels
    print(len(all_preds), len(all_labels))
    # Save the probabilities and labels
    np.save('data/graph/all_probs_model_2.npy', all_probs)
    np.save('data/graph/all_labels_model_2.npy', all_labels)
    np.save('data/graph/precision_model_2.npy', precision)
    np.save('data/graph/recall_model_2.npy', recall)
    np.save('data/graph/pr_thresholds_model_2.npy', pr_thresholds)

    return acc, CM, accuracies, (fpr, tpr, thresholds)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define the device 
model = AE_SUPERVISED().to(device) # Define the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Define the optimizer
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
epochs = 100
# # DATA
# # Load training data
cols = [ 'Speed', 'Compass', 'Accelerometer_x', 'Accelerometer_y',
        'Accelerometer_z', 'GNSS_latitude', 'GNSS_longitude', 
        'GNSS_altitude', 'Control_throttle', 'Control_steer']
# UNCOMMENT FOR TRAINING
# new_data ='data/training_data_attacked.csv'
# data3 = pd.read_csv(new_data)

# # Exclude columns before normalization  vehicle and label
# data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)

# # Apply normalization technique to numeric columns
# for column in data_numeric.columns:
#     data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()

# # Insert 'Vehicle' column as the first column (position 0)
# data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
# data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)

# # Split into groups with labels
# groups = []
# for i in range(0, len(data_evaluation), 25):
#     groups.append(data_evaluation.iloc[i:i + 25])

# train_groups, test_groups = train_test_split(groups, test_size=0.2)
# x_train, labels_train = create_x_y_data(train_groups)
# x_test, labels_test = create_x_y_data(test_groups)
# train_data = TensorDataset(x_train, labels_train)
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
# test_data = TensorDataset(x_test, labels_test)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# # Initialize lists to store losses
# train_losses = []
# test_losses = []
# criterion = nn.CrossEntropyLoss()
# for epoch in range(epochs):
#     model.train()  # Switch to training mode
#     total_train_loss = 0

#     for batch, labels in train_loader:  # Assuming you have a DataLoader

#         batch = batch.view(-1, 200).to(device)
#         labels = labels.long().to(device)

#         optimizer.zero_grad()

#         outputs = model(batch)  # compute outputs
#         # print('labels', labels)
#         # print('output', outputs)
#         # Compute the classification loss
#         loss = nn.CrossEntropyLoss()(outputs, labels)

#         # Update model weights
#         loss.backward()
#         optimizer.step()
#         total_train_loss += loss.item()
#     print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, epochs,total_train_loss / len(x_train)))  # display the epoch training loss
#     train_losses.append(total_train_loss / len(x_train))  # Append train loss

#     model.eval()  # Set the model to evaluation mode

#     test_loss = 0  # initialize test loss
#     with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
#         for batch, labels in test_loader:  # Assuming you have a DataLoader

#             batch = batch.view(-1, 200).to(device)
#             labels = labels.long().to(device)

#             optimizer.zero_grad()

#             outputs = model(batch)  # compute outputs
#             # Compute the classification loss
#             loss = nn.CrossEntropyLoss()(outputs, labels)
#             test_loss += loss.item()

#     test_loss = test_loss / len(x_test)
#     print(f'Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss}\n')
#     # Append test loss
#     test_losses.append(test_loss)

#     # Save model after each epoch
#     torch.save(model, f"data/model_epochs/model_epochs_2/model_2_normal_at_epoch_{epoch}.pt")

# # After all epochs are done, plot the losses
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(test_losses, label='Test Loss')
# plt.title('Epoch vs Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # END OF TRAINING
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # Load the model at best epoch -> epoch is selected manually 
model_at_the_best_epoch = torch.load("data/model_epochs/model_epochs_2/model_2_normal_at_epoch_99.pt",
                                     map_location=torch.device('cpu'))
 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# #            EVALUATION OF THE MODEL                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GENERAL EVALUATION WITH THE SAME TESTING DATASET AS THE FIRST MODEL!
testing_data = 'data/testing_data_for_both_models.csv' 
data3 = pd.read_csv(testing_data)

# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)

# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()

# Insert 'Vehicle' column as the first column (position 0)
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])

# Concatenate 'Label' column as the last column
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)

# Split into groups with labels
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
print('GENERAL EVALUATION')
# uncomment for the further data and uncomment also the other test_model function
# acc_model_2, CM_model_2, accuracies_model_2 = test_model_pr_roc(model_at_the_best_epoch, train_loader, device)
acc_model, CM_model, accuracies_model, roc_data = test_model_pr_roc(model_at_the_best_epoch, train_loader, device)

# roc_data contains the fpr, tpr, and thresholds
fpr, tpr, thresholds = roc_data
np.save('fpr_model_2.npy', fpr)
np.save('tpr_model_2.npy', tpr) 

def save_roc_data(fpr, tpr, thresholds, file_name):
    roc_df = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Thresholds': thresholds
    })
    roc_df.to_csv(file_name, index=False)
save_roc_data(*roc_data, 'roc_data.csv')

#  Uncomment to plot ROC curve
def plot_roc_curve(fpr, tpr, label=None):
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate') 
    if label is not None:
        plt.legend(loc="lower right")
    plt.show()

# Call the plot function with the returned data
plot_roc_curve(fpr, tpr)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # CLOAK ATTACK LOW
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

new_data = 'data/attacks_data/cloak_attack_low.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
print('CLOAK ATTACK LOW')
test_model(model_at_the_best_epoch, train_loader, device)

 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # CLOAK ATTACK MEDIUM
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

new_data = 'data/attacks_data/cloak_attack_medium.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
print('CLOAK ATTACK MEDIUM')
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_model(model_at_the_best_epoch, train_loader, device)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # CLOAK ATTACK HIGH
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

new_data = 'data/attacks_data/cloak_attack_high.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
print('CLOAK ATTACK HIGH')
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_model(model_at_the_best_epoch, train_loader, device)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # FLIP ATTACK LOW 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

new_data = 'data/attacks_data/flip_attack_low.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
print('FLIP ATTACK LOW')
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_model(model_at_the_best_epoch, train_loader, device)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # FLIP ATTACK MEDIUM 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

new_data = 'data/attacks_data/flip_attack_medium.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
print('FLIP ATTACK MEDIUM')
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_model(model_at_the_best_epoch, train_loader, device)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # FLIP ATTACK HIGH 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

new_data = 'data/attacks_data/flip_attack_high.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
print('FLIP ATTACK HIGH')
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_model(model_at_the_best_epoch, train_loader, device)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # GAUSSIAN ATTACK LOW 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

new_data = 'data/attacks_data/gaussian_attack_low.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
print('GAUSSIAN ATTACK LOW')
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_model(model_at_the_best_epoch, train_loader, device)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # GAUSSIAN ATTACK MEDIUM 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

new_data = 'data/attacks_data/gaussian_attack_medium.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
print('GAUSSIAN ATTACK MEDIUM')
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_model(model_at_the_best_epoch, train_loader, device)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # GAUSSIAN ATTACK HIGH 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

new_data = 'data/attacks_data/gaussian_attack_high.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
print('GAUSSIAN ATTACK HIGH')
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_model(model_at_the_best_epoch, train_loader, device)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # REPLAY ATTACK LOW
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

new_data = 'data/attacks_data/replay_attack_low.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
print('REPLAY ATTACK LOW')
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_model(model_at_the_best_epoch, train_loader, device)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # REPLAY ATTACK MEDIUM
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

new_data = 'data/attacks_data/replay_attack_medium.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
print('REPLAY ATTACK MEDIUM')
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_model(model_at_the_best_epoch, train_loader, device)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # REPLAY ATTACK HIGH
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

new_data = 'data/attacks_data/replay_attack_high.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
print('REPLAY ATTACK HIGH')
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_model(model_at_the_best_epoch, train_loader, device)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # SENSOR FAILURE ATTACK LOW
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
new_data = 'data/attacks_data/zero_out_same_intervals_low.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
print('SENSOR FAILURE ATTACK LOW')
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_model(model_at_the_best_epoch, train_loader, device)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # SENSOR FAILURE ATTACK MEDIUM
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
new_data = 'data/attacks_data/zero_out_same_intervals_medium.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
print('SENSOR FAILURE ATTACK MEDIUM')
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 

train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_model(model_at_the_best_epoch, train_loader, device)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # SENSOR FAILURE ATTACK HIGH
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
new_data = 'data/attacks_data/zero_out_same_intervals_high.csv'
 
# new_data = 'obtained_data/new_data/model_1_testing_attacked_data.csv'
data3 = pd.read_csv(new_data)
# Exclude columns before normalization  vehicle and label
data_numeric = data3.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', data3['Vehicle'])
data_evaluation = pd.concat([data_numeric, data3['Label']], axis=1)
print('SENSOR FAILURE ATTACK HIGH')
groups = []
for i in range(0, len(data_evaluation), 25):
    groups.append(data_evaluation.iloc[i:i + 25])

x_train, labels_train = create_x_y_data(groups)
 
train_data = TensorDataset(x_train, labels_train)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
test_model(model_at_the_best_epoch, train_loader, device)