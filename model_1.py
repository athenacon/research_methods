
# Model 1: Autoencoder for 20 inputs and 5 outputs
# The model is trained on normal data and then tested on normal and attacked data
# Predict the future state of the car according to normal unaffected input data
import torch.nn as nn
import torch
import torchvision
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=64
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=64, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=50
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

def create_x_y_data(grouped_data):
    x_data = []
    y_data = []

    for group in grouped_data:

        # Split the data into 'x' and 'y' based on the 'Vehicle' column
        x_temp = group[group['Vehicle'] == 'x'][cols]
        y_temp = group[group['Vehicle'] == 'y'][cols]

        if not x_temp.empty and len(group) == 25:
            x_data.append(torch.from_numpy(x_temp.values.flatten()).float())

        if not y_temp.empty and len(group) == 25:
            y_data.append(torch.from_numpy(y_temp.values.flatten()).float())

    return x_data, y_data


def test_model_pr_curce(model_at_the_best_epoch, dataloaders, device):
    # https://yeseullee0311.medium.com/pytorch-performance-evaluation-of-a-classification-model-confusion-matrix-fbec6f4e8d0
    model_at_the_best_epoch.eval()
    CM = np.zeros((2, 2))

    # Split into groups with labels
    groups = []
    for i in range(0, len(dataloaders), 25):
        groups.append(dataloaders.iloc[i:i + 25])

    x_test, y_test, labels_train = create_x_y_label_data(groups)
    rec = []
    accuracies = []
    all_reconstruction_errors = []
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch_features, batch_labels, attack_labels in zip(x_test, y_test, labels_train):
            batch_features = batch_features.view(-1, 200).to(device)
            batch_labels = batch_labels.view(-1, 50).to(device)

            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model_at_the_best_epoch(batch_features)
            reconstruction_errors = criterion(outputs, batch_labels)
            rec.append(reconstruction_errors)
            preds = (reconstruction_errors > 0.176).long()
            # move labels and preds to cpu for confusion matrix
            attack_labels = attack_labels.view(-1).cpu()
            preds = preds.view(-1).cpu()
            CM += confusion_matrix(attack_labels.numpy(), preds.numpy(), labels=[0,1])
            batch_acc = (torch.sum(preds == attack_labels).item()) / attack_labels.size(0)
            accuracies.append(batch_acc)

            rec_errors = reconstruction_errors.cpu().numpy()
            rec_errors = rec_errors if rec_errors.ndim != 0 else [rec_errors.item()]
            all_reconstruction_errors.extend(rec_errors)
            all_labels.extend(attack_labels.numpy())
            
            # chi square 
            all_preds.extend(preds.cpu().numpy())
    # chi square 
    np.save('data/graph/all_preds_model_1.npy', all_preds)       # Predictions
    np.save('data/graph/all_labels_model_1.npy', all_labels)     # True labels
    
    # print('dev', all_preds)
    # print('dev', all_labels)
    # print('dev', len(all_preds), len(all_labels))

    tn=CM[0][0]
    tp=CM[1][1]
    fp=CM[0][1]
    fn=CM[1][0]
    acc=np.sum(np.diag(CM)/np.sum(CM))
    sensitivity=tp/(tp+fn)
    precision=tp/(tp+fp)
    
    print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
    print()
    print('Confusion Matirx : ')
    print(CM)
    print('- Sensitivity : ',(tp/(tp+fn))*100)
    print('- Specificity : ',(tn/(tn+fp))*100)
    print('- Precision: ',(tp/(tp+fp))*100)
    print('- NPV: ',(tn/(tn+fn))*100)
    print('- F1 : ',((2*sensitivity*precision)/(sensitivity+precision))*100)
    print()
    precision, recall, pr_thresholds = precision_recall_curve(all_labels, all_reconstruction_errors)

    # Save precision and recall values
    np.save('precision_model_1.npy', precision)
    np.save('recall_model_1.npy', recall)
    np.save('pr_thresholds_model_1.npy', pr_thresholds)
    # save for chi square test
    np.save('data/graph/model_1_predictions_chi_square.npy', preds.numpy())  # Saving the predictions
    np.save('data/graph/model_1_true_labels_chi_square.npy', attack_labels.numpy())  # Saving the true labels
    print(preds.shape)
    # uncomment to save the reconstruction errors for chi square test
    # return acc, CM, accuracies  
    return acc, CM, accuracies, np.array(all_reconstruction_errors), np.array(all_labels)
    # return acc, CM, accuracies
     
def test_model(model_at_the_best_epoch, dataloaders, device):
    # https://yeseullee0311.medium.com/pytorch-performance-evaluation-of-a-classification-model-confusion-matrix-fbec6f4e8d0
    model_at_the_best_epoch.eval()
    CM = np.zeros((2, 2))

    # Split into groups with labels
    groups = []
    for i in range(0, len(dataloaders), 25):
        groups.append(dataloaders.iloc[i:i + 25])

    x_test, y_test, labels_train = create_x_y_label_data(groups)
    rec = []
    accuracies = []
    all_reconstruction_errors = []
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch_features, batch_labels, attack_labels in zip(x_test, y_test, labels_train):
            batch_features = batch_features.view(-1, 200).to(device)
            batch_labels = batch_labels.view(-1, 50).to(device)

            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model_at_the_best_epoch(batch_features)
            reconstruction_errors = criterion(outputs, batch_labels)
            rec.append(reconstruction_errors)
            preds = (reconstruction_errors > 0.176).long()
            # move labels and preds to cpu for confusion matrix
            attack_labels = attack_labels.view(-1).cpu()
            preds = preds.view(-1).cpu()
            CM += confusion_matrix(attack_labels.numpy(), preds.numpy(), labels=[0,1])
            batch_acc = (torch.sum(preds == attack_labels).item()) / attack_labels.size(0)
            accuracies.append(batch_acc)

            # rec_errors = reconstruction_errors.cpu().numpy()
            # rec_errors = rec_errors if rec_errors.ndim != 0 else [rec_errors.item()]
            # all_reconstruction_errors.extend(rec_errors)
            # all_labels.extend(attack_labels.numpy())
            
            # # chi square 
            # all_preds.extend(preds.cpu().numpy())
    # # chi square 
    # np.save('data/graph/all_preds_model_1.npy', all_preds)       # Predictions
    # np.save('data/graph/all_labels_model_1.npy', all_labels)     # True labels
    
    # print('dev', all_preds)
    # print('dev', all_labels)
    # print('dev', len(all_preds), len(all_labels))

    tn=CM[0][0]
    tp=CM[1][1]
    fp=CM[0][1]
    fn=CM[1][0]
    acc=np.sum(np.diag(CM)/np.sum(CM))
    sensitivity=tp/(tp+fn)
    precision=tp/(tp+fp)
    
    print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
    print()
    print('Confusion Matirx : ')
    print(CM)
    print('- Sensitivity : ',(tp/(tp+fn))*100)
    print('- Specificity : ',(tn/(tn+fp))*100)
    print('- Precision: ',(tp/(tp+fp))*100)
    print('- NPV: ',(tn/(tn+fn))*100)
    print('- F1 : ',((2*sensitivity*precision)/(sensitivity+precision))*100)
    print()

    return acc, CM, accuracies
     
def create_x_y_label_data(grouped_data):
    x_data = []
    y_data = []
    labels = []

    for group in grouped_data:
        # Split the data into 'x' and 'y' based on the 'Vehicle' column
        x_temp = group[group['Vehicle'] == 'x'][cols]
        y_temp = group[group['Vehicle'] == 'y'][cols]

        # Add labels
        label = group['Label'].iloc[0]  # Take the label of the group
        labels.append(torch.tensor(label).float())

        if not x_temp.empty and len(group) == 25:
            # if not x_temp.empty and len(group) == 10:
            # all x data
            x_data.append(torch.from_numpy(x_temp.values.flatten()).float())

        if not y_temp.empty and len(group) == 25:
            # if not y_temp.empty and len(group) == 10:
            # all y data
            y_data.append(torch.from_numpy(y_temp.values.flatten()).float())

    return x_data, y_data, labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define the device cpu
model = AE(input_shape=200).to(device) # Define the model
criterion = nn.MSELoss() # Define the loss function (Mean squared error)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Define the optimizer

# DATA
# Load training data
data_path1 = 'data/training_data_normal.csv'
data1 = pd.read_csv(data_path1)

# x_data real data 20, y_data to be predicted the next 5
x_data = []
y_data = []

cols = ['Speed', 'Compass', 'Accelerometer_x', 'Accelerometer_y',
        'Accelerometer_z', 'GNSS_latitude', 'GNSS_longitude',
        'GNSS_altitude', 'Control_throttle', 'Control_steer']

# uncomment to retrain the model
# # Exclude non-numeric columns before normalization (we exclude the first column of x and y the 'Vehicle' column)
# data_numeric = data1[cols]

# # Apply normalization technique to numeric columns
# for column in data_numeric.columns:
#     data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()

# # Combine the normalized numeric columns with the non-numeric 'Vehicle' column
# data_normalized = pd.concat([data1['Vehicle'], data_numeric], axis=1)
# # print(data_normalized)
# # Now you have the dataset with normalized numeric columns, you can loop over it:
# groups = []
# for i in range(0, len(data_normalized), 25):
#     groups.append(data_normalized.iloc[i:i + 25])
#     # print(groups)

# train_groups, test_groups = train_test_split(groups, test_size=0.2)

# x_train, y_train = create_x_y_data(train_groups)
# x_test, y_test = create_x_y_data(test_groups)

# # DATA FINISHED
# x_train, y_train = create_x_y_data(train_groups)
# x_test, y_test = create_x_y_data(test_groups)
# x_train = torch.stack(x_train)
# y_train = torch.stack(y_train)
# x_test = torch.stack(x_test)
# y_test = torch.stack(y_test)

# train_losses = []
# test_losses = []
# train_data = TensorDataset(x_train, y_train)
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# test_data = TensorDataset(x_test, y_test)
# test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# print(len(x_train), len(y_train), len(x_test), len(y_test))
# transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# epochs = 100 
# # Initialize lists to store losses
# train_losses = []
# test_losses = []
# # Train the model
# for epoch in range(epochs):
#     loss = 0
#     for batch_features, batch_labels in train_loader:
#         # print(batch_features.shape)
#         # reshape mini-batch data to [N, input_shape] matrix
#         # load it to the active device
#         batch_features = batch_features.view(-1, 200).to(device)
#         # print(epoch, batch_features)
#         batch_labels = batch_labels.view(-1, 50).to(device)

#         # reset the gradients back to zero
#         optimizer.zero_grad()

#         # compute reconstructions
#         outputs = model(batch_features)
#         # print(outputs)
#         # compute training reconstruction loss
#         train_loss = criterion(outputs, batch_labels)

#         # compute accumulated gradients
#         train_loss.backward()

#         # perform parameter update based on current gradients
#         optimizer.step()

#         # add the mini-batch training loss to epoch loss
#         loss += train_loss.item()

#     # compute the epoch training loss
#     loss = loss / len(train_loader)

#     # display the epoch training loss
#     print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
#     # Append train loss
#     train_losses.append(loss)
#     test_loss = 0

#     with torch.no_grad():
#         for test_features, test_labels in test_loader:
#             # reshape mini-batch data to [N, input_shape] matrix
#             # load it to the active device
#             test_features = test_features.view(-1, 200).to(device)
#             test_labels = test_labels.view(-1, 50).to(device)

#             # compute reconstructions
#             outputs = model(test_features)
#             # print(outputs)

#             # compute test reconstruction loss
#             test_loss_temp = criterion(outputs, test_labels)

#             # add the mini-batch test loss to epoch loss
#             test_loss += test_loss_temp.item()

#         # compute the epoch test loss
#         test_loss = test_loss / len(test_loader)

#         # display the epoch test loss
#         print("epoch : {}/{}, test loss = {:.6f}".format(epoch + 1, epochs, test_loss))
#         # Append test loss
#         test_losses.append(test_loss)

#         # Save model after each epoch
#         torch.save(model, f"data/model_epochs/model_epochs_1/model_1_at_epoch_{epoch}.pt")

# END OF TRAINING

# Load the model at best epoch -> epoch is selected manually
model_at_the_best_epoch = torch.load("data/model_epochs/model_epochs_1/model_1_at_epoch_72.pt",
                                     map_location=torch.device('cpu'))
 
 
# After all epochs are done, plot the losses
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(test_losses, label='Test Loss')
# plt.title('Epoch vs Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # DETECTION --> Find the threshold: normal_data & max
# data_path1= 'data/normal_threshold_data_first_model.csv'
# data2 = pd.read_csv(data_path1)
# # Exclude non-numeric columns before normalization (we exclude the first column of x and y the 'Vehicle' column)
# data_numeric = data2[cols]

# # Apply normalization technique to numeric columns
# for column in data_numeric.columns:
#     data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()

# # Combine the normalized numeric columns with the non-numeric 'Vehicle' column
# data_normalized = pd.concat([data2['Vehicle'], data_numeric], axis=1)

# # Now you have the dataset with normalized numeric columns, you can loop over it:
# groups = []
# for i in range(0, len(data_normalized), 25):
#     groups.append(data_normalized.iloc[i:i + 25])
#     # print(groups)

# x_train, y_train = create_x_y_data(groups)
# # print('dev', len(x_train[5]), len(y_train[5]))

# # Keep track of the maximum loss on the non-attacked data
# max_non_attacked_loss = 0
# s=[]
# with torch.no_grad():
#     for batch_features, batch_labels in zip(x_train, y_train):
#         # reshape mini-batch data to [N, input_shape] matrix
#         # load it to the active device
#         batch_features = batch_features.view(-1, 200).to(device)
#         batch_labels = batch_labels.view(-1, 50).to(device)

#         # compute reconstructions
#         outputs = model_at_the_best_epoch(batch_features)
        
#         # loss = torch.norm(batch_labels - outputs).item()

#         # compute training reconstruction loss
#         loss = criterion(outputs, batch_labels).item()
#         s.append(loss)
#         if loss > max_non_attacked_loss:
#             max_non_attacked_loss = loss

# # print(s)
# # Secondly, we load the attacked data
# data_path_attack = 'data/attacked_threshold_data_first_model.csv'
# data_attacked_threshold_minimum = pd.read_csv(data_path_attack)

# # Exclude non-numeric columns before normalization (we exclude the first column of x and y the 'Vehicle' column)
# data_numeric = data_attacked_threshold_minimum[cols]

# # Apply normalization technique to numeric columns
# for column in data_numeric.columns:
#     data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()

# # data_numeric.insert(0, 'Vehicle', data_attacked_threshold_minimum['Vehicle'])
# data_normalized = pd.concat([data_attacked_threshold_minimum['Vehicle'], data_numeric], axis=1)

# groups_att = []
# for i in range(0, len(data_normalized), 25):
#     groups_att.append(data_normalized.iloc[i:i + 25])
#     # print(groups)

# # group = data_normalized.iloc[i*25 : (i+1)*25]
# x_train_attacked, y_train_attacked = create_x_y_data(groups_att)

# # print('dev', len(x_train_attacked), len(y_train_attacked))
# # print('dev', len(x_train_attacked[0]), len(y_train_attacked[0]))
# # Now calculate the minimum loss on the attacked data
# min_attacked_loss = float('inf')
# d=[]
# with torch.no_grad():
#     for batch_features, batch_labels in zip(x_train_attacked, y_train_attacked):
#         # reshape mini-batch data to [N, input_shape] matrix
#         # load it to the active device
#         batch_features = batch_features.view(-1, 200).to(device)
#         batch_labels = batch_labels.view(-1, 50).to(device)

#         # compute reconstructions
#         outputs = model_at_the_best_epoch(batch_features)
#         # compute training reconstruction loss
#         # loss = torch.norm(batch_labels - outputs).item()
#         loss = criterion(outputs, batch_labels).item()
#         d.append(loss)
#         # compute training reconstruction loss
#         # loss = criterion(outputs, batch_labels).item()
#         if loss < min_attacked_loss:
#             min_attacked_loss = loss

# # Finally, we set your threshold as the average of these two values
# threshold = (max_non_attacked_loss + min_attacked_loss) / 2

# print(threshold, "max", max_non_attacked_loss, "min", min_attacked_loss)
# # #    DETECTOR IMPLEMENTATION FINISHED
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# #            EVALUATION OF THE MODEL                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# GENERAL EVALUATION WITH THE SAME TESTING DATASET AS THE FIRST MODEL!
testing_data = 'data/testing_data_for_both_models.csv'
testing_evaluation = pd.read_csv(testing_data)
# Exclude columns before normalization  vehicle and label
data_numeric = testing_evaluation.drop(['Vehicle', 'Label'], axis=1)
# Apply normalization technique to numeric columns
for column in data_numeric.columns:
    data_numeric[column] = data_numeric[column] / data_numeric[column].abs().max()
data_numeric.insert(0, 'Vehicle', testing_evaluation['Vehicle'])
data_evaluation = pd.concat([data_numeric, testing_evaluation['Label']], axis=1)
print('GENERAL EVALUATION')
# Uncomment the following line to test the model on the testing dataset and get the data for ROC / PR curves
# test_model(model_at_the_best_epoch, data_evaluation, device)
# # accuracy, confusion_matrix, accuracies, reconstruction_errors, true_labels= test_model_pr_curce(model_at_the_best_epoch, data_evaluation, device)
# # np.save('accuracies_model_1.npy', accuracies)

# # Compute ROC curve and ROC area for each class
# fpr, tpr, thresholds = roc_curve(true_labels, reconstruction_errors)
# roc_auc = auc(fpr, tpr)
 
# np.save('fpr_model_1.npy', fpr)
# np.save('tpr_model_1.npy', tpr)
# np.save('thresholds_model_1.npy', thresholds)
# np.save('roc_auc_model_1.npy', np.array([roc_auc])) 
# # # Plot ROC curve
# plt.figure()
# lw = 2 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# EVALUATION BASED ON INDIVIDUAL ATTACKS AND THEIR CORRESPONDING LEVEL
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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
print('CLOAK ATTACK LOW')
test_model(model_at_the_best_epoch, data_evaluation, device)

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
test_model(model_at_the_best_epoch, data_evaluation, device)

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
test_model(model_at_the_best_epoch, data_evaluation, device)


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
test_model(model_at_the_best_epoch, data_evaluation, device)


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
test_model(model_at_the_best_epoch, data_evaluation, device)


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
test_model(model_at_the_best_epoch, data_evaluation, device)

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
test_model(model_at_the_best_epoch, data_evaluation, device)

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
test_model(model_at_the_best_epoch, data_evaluation, device)

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
test_model(model_at_the_best_epoch, data_evaluation, device)


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
test_model(model_at_the_best_epoch, data_evaluation, device)


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
test_model(model_at_the_best_epoch, data_evaluation, device)

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
test_model(model_at_the_best_epoch, data_evaluation, device)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# SENSOR FAILURE ATTACK LOW
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
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
test_model(model_at_the_best_epoch, data_evaluation, device)

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
test_model(model_at_the_best_epoch, data_evaluation, device)

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
test_model(model_at_the_best_epoch, data_evaluation, device) 