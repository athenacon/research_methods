import numpy as np
import random
# def flip_attack(group):
#     # Make a copy of the group to avoid changing the original dataframe
#     group_copy = group.copy()
    
#     # Randomly select either 'gnsslatitude' or 'gnsslongitude'
#     column_to_flip = np.random.choice(['GNSS_longitude', 'GNSS_latitude'])

#     # Multiply the selected column by -1
#     group_copy[column_to_flip] = group_copy[column_to_flip] * -1

#     return group_copy

def flip_attack(group, subset_size, columns=['GNSS_latitude', 'GNSS_longitude']):
    '''
    Perform flip attack on a random subset of data within the given group
    
    group: pandas DataFrame representing a single group
    subset_size: Proportion of the group to attack, represented as a float between 0 and 1
    columns: Columns to potentially flip (randomly selected)
    '''
    
    # Create a copy of the group to avoid modifying the original DataFrame
    attacked_group = group.copy()
    
    # Choose random column from columns
    random_column = random.choice(columns)
    
    # Calculate the number of rows to select for the subset based on the provided proportion
    num_rows = int(len(group) * subset_size)
    
    # Choose random subset within the group
    subset = group.sample(num_rows)
    
    # Flip the values in the subset for the selected column
    subset[random_column] = subset[random_column] * -1
    
    # Update the values in the original DataFrame
    attacked_group.update(subset)

    return attacked_group
