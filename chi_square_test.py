import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from matplotlib.patches import Patch
import scipy
print("SciPy version:", scipy.__version__)
 
# Step 1: Load the saved data
model_1_preds = np.load('data/graph/all_preds_model_1.npy')
model_1_true = np.load('data/graph/all_labels_model_1.npy')

model_2_preds = np.load('data/graph/all_preds_model_2.npy')
model_2_true = np.load('data/graph/all_labels_model_2.npy')

# print(model_1_preds)
print("Shape of model_1_preds:", np.shape(model_1_preds))
print("Shape of model_2_preds:", np.shape(model_2_preds))

# Step 2: Check if the true labels are the same
assert np.array_equal(model_1_true, model_2_true), "True labels do not match between models."

# Step 3: Create the contingency table
contingency_table = np.zeros((2, 2))
contingency_table[0, 0] = np.sum((model_1_preds == model_1_true) & (model_2_preds == model_2_true))
contingency_table[0, 1] = np.sum((model_1_preds != model_1_true) & (model_2_preds == model_2_true))
contingency_table[1, 0] = np.sum((model_1_preds == model_1_true) & (model_2_preds != model_2_true))
contingency_table[1, 1] = np.sum((model_1_preds != model_1_true) & (model_2_preds != model_2_true))
 
res = chi2_contingency(contingency_table)
print(res.statistic)
print(res.pvalue)

chi2, p, dof, expected = chi2_contingency(contingency_table)
 
print("Contingency Table:")
print(contingency_table)
total = np.sum(contingency_table)
percentage_contingency_table = (contingency_table / total) * 100
print("Percentage Contingency Table:")
formatted_percentage_table = np.array2string(percentage_contingency_table)
print(formatted_percentage_table)
 
# Create a dictionary for the mosaic plot
# The keys are tuples, where the first element is Model 1's correctness and the second is Model 2's
data = {('Model 1 Correct', 'Model 2 Correct'): contingency_table[0, 0],
        ('Model 1 Incorrect', 'Model 2 Correct'): contingency_table[0, 1],
        ('Model 1 Correct', 'Model 2 Incorrect'): contingency_table[1, 0],
        ('Model 1 Incorrect', 'Model 2 Incorrect'): contingency_table[1, 1]}

# Define properties (colors) for each category
color_dict = {
    ('Model 1 Correct', 'Model 2 Correct'): 'lightgreen',
    ('Model 1 Incorrect', 'Model 2 Correct'): 'lightblue',
    ('Model 1 Correct', 'Model 2 Incorrect'): 'lightcoral',
    ('Model 1 Incorrect', 'Model 2 Incorrect'): 'lightsalmon'
}

# # Create the mosaic plot
# # mosaic(data, properties=lambda key: {'color': color_dict[key]})
# plt.title('Mosaic Plot of Contingency Table')

# # Create a custom legend
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='lightgreen', label='Model 1 Correct, Model 2 Correct'),
#     Patch(facecolor='lightblue', label='Model 1 Incorrect, Model 2 Correct'),
#     Patch(facecolor='lightcoral', label='Model 1 Correct, Model 2 Incorrect'),
#     Patch(facecolor='lightsalmon', label='Model 1 Incorrect, Model 2 Incorrect')
# ]
# plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.show()
# Adjusting figure size to accommodate legend
# plt.figure(figsize=(10, 10))

# Create the mosaic plot
mosaic(data, properties=lambda key: {'color': color_dict[key]})
plt.title('Mosaic Plot of Contingency Table')

# Create a custom legend
legend_elements = [
    Patch(facecolor='lightgreen', label='Model 1 Correct, Model 2 Correct'),
    Patch(facecolor='lightblue', label='Model 1 Incorrect, Model 2 Correct'),
    Patch(facecolor='lightcoral', label='Model 1 Correct, Model 2 Incorrect'),
    Patch(facecolor='lightsalmon', label='Model 1 Incorrect, Model 2 Incorrect')
]

# Positioning the legend
plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(1, 1))
plt.show()