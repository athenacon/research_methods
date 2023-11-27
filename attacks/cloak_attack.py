import numpy as np
def zero_out_attack(data, columns_to_attack):
    """ Zero out attack.
        Returns the attacked data.

        data: The dataframe
        columns_to_attack: The columns that will be attacked
    """
    # Make a copy of the data
    # new_data = data.copy()
    new_data = data.copy().reset_index(drop=True)

    for column_to_attack in columns_to_attack:
        # Randomly select a row in the first 5% of the data
        start_idx = np.random.randint(0, int(0.05 * len(new_data[column_to_attack])))

        # Get the value in the selected row
        value = new_data.loc[start_idx, column_to_attack]

        # Start from the selected row and decrease the values linearly or non-linearly
        for idx in range(start_idx, len(new_data[column_to_attack])):
            # Get the decrease rate
            decrease_rate = np.random.uniform(0.9, 1.4)
            # Decrease the value linearly or non-linearly based on the value in the selected row
            new_data.loc[idx, column_to_attack] = value * (1 - decrease_rate) ** (idx - start_idx)

            # If the current value is close to zero, set all subsequent values to zero
            if abs(new_data.loc[idx, column_to_attack]) < 0.001:
                new_data.loc[idx:, column_to_attack] = 0
                break

    return new_data

# def zero_out_attack(data, columns_to_attack):
#     """ Zero out attack.
#         Returns the attacked data.
#
#         data: The dataframe
#         columns_to_attack: The columns that will be attacked
#     """
#     # Make a copy of the data and reset the index
#     new_data = data.copy().reset_index(drop=True)
#
#     for column_to_attack in columns_to_attack:
#         # Randomly select a row in the first 5% of the data
#         start_idx = np.random.randint(0, int(0.05 * len(new_data[column_to_attack])))
#
#         # Get the value in the selected row
#         value = new_data.loc[start_idx, column_to_attack]
#
#         # Start from the selected row and decrease the values linearly or non-linearly
#         for idx in range(start_idx, len(new_data[column_to_attack])):
#             # Get the decrease rate
#             decrease_rate = np.random.uniform(0.1, 0.4)
#             # Decrease the value linearly or non-linearly based on the value in the selected row
#             new_data.loc[idx, column_to_attack] = value * (1 - decrease_rate) ** (idx - start_idx)
#
#             # If the current value is close to zero, set all subsequent values to zero
#             if abs(new_data.loc[idx, column_to_attack]) < 0.001:
#                 new_data.loc[idx:, column_to_attack] = 0
#                 break
#
#     return new_data
