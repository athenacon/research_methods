import numpy as np
import random

# GAUSSIAN ATTACK: 
# Select manually or randomly the data you want to amend. 
# Amend all the elements of the data or a proportion of them with intervals or purely random.

def gaussian_attack(data, columns_to_attack, std_dev):
    """
    Implements a Gaussian attack on the specified columns of the given dataset.

    Args:
        data (pandas.DataFrame): The input dataset to be attacked.
        columns_to_attack (list): A list of column names to be attacked.
        std_dev (float): Standard deviation of the Gaussian noise to be added.

    Returns:
        pandas.DataFrame: The attacked dataset.

    """
    mean = 0

    for col in columns_to_attack:
        prop = random.uniform(0.99, 1)
        # prop = 1
        # Determine the number of elements to add noise to
        num_noisy = int(prop * len(data[col]))  # Example for prop = 0.5 and len(data[col]) = 740: num_noisy = 370

        # Generate a boolean mask of the same length as the column, with 'num_noisy' True values
        mask = np.full(len(data[col]), False)
        mask[:num_noisy] = True
        np.random.shuffle(mask)

        # Generate the noise
        noise = np.random.normal(mean, std_dev, num_noisy)

        # Add the noise to the selected elements
        data.loc[mask, col] += noise

    return data
