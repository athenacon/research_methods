import numpy as np

def zero_out_random_intervals(data, cols):
    """Implements an attack that zeroes out random intervals of data in specified columns.
    Returns the attacked data.
    
    data: The original data
    cols: The columns to perform the attack on
    prop: The proportion of data in each column to zero out
    """

    for col in cols:
        prop = np.random.uniform(0.99, 1)
        # Determine the number of elements to zero out
        num_zeroed = int(prop * len(data))
        # Create a copy of the column data
        col_data = data[col].copy()

        while num_zeroed > 0:
            # Choose a random start point
            start = np.random.randint(0, len(data))

            # Choose a random interval length
            interval = np.random.randint(1, num_zeroed + 1)

            # Zero out the elements in the interval
            col_data[start:start + interval] = 0

            # Subtract the interval length from the number of elements to zero out
            num_zeroed -= interval

        # Replace the column in the dataframe
        data[col] = col_data

    return data

# def zero_out_same_intervals(data, cols):
#     """Implements an attack that zeroes out the same intervals of data across specified columns.
#     Returns the attacked data.
    
#     data: The original data
#     cols: The columns to perform the attack on
#     prop: The proportion of data in each column to zero out
#     """
#     # prop = np.random.uniform(0.99, 1)
#     prop = 1
#     # Determine the number of elements to zero out
#     num_zeroed = int(prop * len(data))

#     # Define intervals to zero out
#     intervals = []

#     while num_zeroed > 0:
#         # Choose a random start point
#         start = np.random.randint(0, len(data)/3)

#         # Choose a random interval length
#         interval = np.random.randint(1, num_zeroed + 1)

#         # Append the interval to the list
#         intervals.append((start, start + interval))

#         # Subtract the interval length from the number of elements to zero out
#         num_zeroed -= interval

#     for col in cols:
#         # Create a copy of the column data
#         col_data = data[col].copy()

#         # Zero out the elements in the intervals
#         for start, end in intervals:
#             col_data[start:end] = 0

#         # Replace the column in the dataframe
#         data[col] = col_data

#     return data

def zero_out_same_intervals(data, cols):
    """Implements an attack that zeroes out the same intervals of data across specified columns.
    Returns the attacked data.
    
    data: The original data
    cols: The columns to perform the attack on
    """
    start = 0
    end = len(data)

    for col in cols:
        # Create a copy of the column data
        col_data = data[col].copy()

        # Zero out the elements in the intervals
        col_data[start:end] = 0

        # Replace the column in the dataframe
        data[col] = col_data

    return data
