import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# Load the ROC data from disk for the first model
fpr1 = np.load('fpr_model_1.npy')
tpr1 = np.load('tpr_model_1.npy')
roc_auc1 = auc(fpr1, tpr1)

# Assume we have the second set of ROC data for another model
fpr2 = np.load('fpr_model_2.npy')
tpr2 = np.load('tpr_model_2.npy')
roc_auc2 = auc(fpr2, tpr2)

# Function to plot ROC Curve for two models
def plot_combined_roc_curve(fpr1, tpr1, roc_auc1, fpr2, tpr2, roc_auc2):
    plt.figure()
    plt.plot(fpr1, tpr1, color='darkorange', lw=2, label=f'Model 1 ROC curve (area = {roc_auc1:.2f})')
    plt.plot(fpr2, tpr2, lw=2, label=f'Model 2 ROC curve (area = {roc_auc2:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

# Call the plot function with the loaded data
plot_combined_roc_curve(fpr1, tpr1, roc_auc1, fpr2, tpr2, roc_auc2)

# Load precision and recall for the first model
precision_model_1 = np.load('precision_model_1.npy')
recall_model_1 = np.load('recall_model_1.npy')
# Calculate the AUC for the first model
pr_auc_model_1 = auc(recall_model_1, precision_model_1)

all_probs_model_2 = np.load('data/graph/all_probs_model_2.npy')
all_labels_model_2 = np.load('data/graph/all_labels_model_2.npy')
precision_model_2 = np.load('data/graph/precision_model_2.npy')
recall_model_2 = np.load('data/graph/recall_model_2.npy')
pr_auc_model_2 = auc(recall_model_2, precision_model_2)
# Now plot both models' precision-recall curves on the same plot
plt.figure()  # You can set the figure size as you prefer

# Plot for Model 1
plt.plot(recall_model_1,  precision_model_1, lw=2, color='darkorange', label=f'Model 1 Precision-Recall curve (area = {pr_auc_model_1:.2f})')

# Plot for Model 2
plt.plot(recall_model_2, precision_model_2,lw=2, label=f'Model 2 Precision-Recall curve (area = {pr_auc_model_2:.2f})')

# Labels and title
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Comparison of Precision-Recall Curves')
plt.legend(loc="best")

# Display the combined plot
plt.show()