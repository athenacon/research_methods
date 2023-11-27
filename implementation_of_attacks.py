import torch.nn as nn
import torch
import torchvision
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import random
import matplotlib.pyplot as plt
from attacks.gaussian_attack import gaussian_attack
from attacks.salt_and_pepper_attack import salt_and_pepper_attack
from attacks.replay_attack import replay_attack
from attacks.flip_attack import flip_attack
from attacks.sensor_failure_attack import zero_out_random_intervals, zero_out_same_intervals
from attacks.cloak_attack import zero_out_attack

cols = ['Speed', 'Compass', 'Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'GNSS_latitude', 
        'GNSS_longitude', 'GNSS_altitude', 'Control_throttle', 'Control_steer']

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CLOAK ATTACK LOW
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'data/models_testing_before_attacks.csv'

data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)

# Attack functions

cols_of_att = ['Speed', 'Compass', 'Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'GNSS_latitude', 
                'GNSS_longitude', 'GNSS_altitude', 'Control_throttle', 'Control_steer']


# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
        random_number = np.random.randint(1, 4)

        # Select a random column(s) to attack based on the random number
        column_to_attack = random.sample(cols, random_number)

        # Apply the attack
        group = zero_out_attack(group, column_to_attack)

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/cloak_attack_low.csv', index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CLOAK ATTACK LOW FINISHED
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CLOAK ATTACK MEDIUM FINISHED
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'data/models_testing_before_attacks.csv'
data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)

# Attack functions

cols_of_att = ['Speed', 'Compass', 'Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'GNSS_latitude', 
                'GNSS_longitude', 'GNSS_altitude', 'Control_throttle', 'Control_steer']


# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
        random_number = np.random.randint(4, 7)

        # Select a random column(s) to attack based on the random number
        column_to_attack = random.sample(cols, random_number)

        # Apply the attack
        group = zero_out_attack(group, column_to_attack)

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/cloak_attack_medium.csv', index=False)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CLOAK ATTACK MEDIUM 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CLOAK ATTACK HIGH FINISHED
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'data/models_testing_before_attacks.csv'
data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)

# Attack functions

cols_of_att = ['Speed', 'Compass', 'Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'GNSS_latitude', 
                'GNSS_longitude', 'GNSS_altitude', 'Control_throttle', 'Control_steer']


# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
        random_number = np.random.randint(7, 11)

        # Select a random column(s) to attack based on the random number
        column_to_attack = random.sample(cols, random_number)

        # Apply the attack
        group = zero_out_attack(group, column_to_attack)

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/cloak_attack_high.csv', index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# CLOAK ATTACK HIGH FINISHED
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FLIP ATTACK LOW 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'data/models_testing_before_attacks.csv'
data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)

# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
        # Apply the attack
        # Define proportion of the group to attack
        group = flip_attack(group, random.uniform(0.1, 0.4))# Attack 10% to 30% of the group

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/flip_attack_low.csv', index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FLIP ATTACK LOW FINISHED
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FLIP ATTACK MEDIUM 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'models_1_testing_2.csv'


data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)

# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
        # Apply the attack
        # Define proportion of the group to attack
        group = flip_attack(group, random.uniform(0.4, 0.7))# Attack 10% to 30% of the group

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/flip_attack_medium.csv', index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FLIP ATTACK MEDIUM FINISHED
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FLIP ATTACK HIGH 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'data/models_testing_before_attacks.csv'
data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)

# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
        # Apply the attack
        # Define proportion of the group to attack
        group = flip_attack(group, random.uniform(0.7, 0.91))# Attack 10% to 30% of the group

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/flip_attack_high.csv', index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FLIP ATTACK HIGH FINISHED
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GAUSSIAN ATTACK LOW 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'data/models_testing_before_attacks.csv' 

data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)
random_number_of_cols = np.random.randint(1, 3)
cols_att = random.sample(cols_of_att, random_number_of_cols)
# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
   
        # Define proportion of the group to attack
        group = gaussian_attack(group, cols_att, 1 ) 

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/gaussian_attack_low.csv', index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GAUSSIAN ATTACK LOW FINISHED
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GAUSSIAN ATTACK MEDIUM 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'data/models_testing_before_attacks.csv'
data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)
random_number_of_cols = np.random.randint(4, 7)
cols_att = random.sample(cols_of_att, random_number_of_cols)
# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
   
        # Define proportion of the group to attack
        group = gaussian_attack(group, cols_att, 4)  

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/gaussian_attack_medium.csv', index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GAUSSIAN ATTACK MEDIUM FINISHED
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GAUSSIAN ATTACK HIGH 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'data/models_testing_before_attacks.csv'
data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)
random_number_of_cols = np.random.randint(7,11)
cols_att = random.sample(cols_of_att, random_number_of_cols)
# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
   
        # Define proportion of the group to attack
        group = gaussian_attack(group, cols_att,  9) 

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/gaussian_attack_high.csv', index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GAUSSIAN ATTACK HIGH FINISHED
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# REPLAY ATTACK LOW
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'data/models_testing_before_attacks.csv'
data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)

# Attack functions

cols_of_att = ['Speed', 'Compass', 'Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'GNSS_latitude', 
                'GNSS_longitude', 'GNSS_altitude', 'Control_throttle', 'Control_steer']

# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
        
        # Apply the attack
        group = replay_attack(group, cols, np.random.randint(10,40) / 100)

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/replay_attack_low.csv', index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# REPLAY ATTACK LOW FINISHED
# # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# REPLAY ATTACK MEDIUM
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'data/models_testing_before_attacks.csv'
data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)

# Attack functions

cols_of_att = ['Speed', 'Compass', 'Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'GNSS_latitude', 
                'GNSS_longitude', 'GNSS_altitude', 'Control_throttle', 'Control_steer']

# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
        
        # Apply the attack
        group = replay_attack(group, cols, np.random.randint(40, 70) / 100)

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/replay_attack_medium.csv', index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# REPLAY ATTACK MEDIUM FINISHED
# # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# REPLAY ATTACK HIGH
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'data/models_testing_before_attacks.csv'
data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)

# Attack functions

cols_of_att = ['Speed', 'Compass', 'Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'GNSS_latitude', 
                'GNSS_longitude', 'GNSS_altitude', 'Control_throttle', 'Control_steer']

# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
        
        # Apply the attack
        group = replay_attack(group, cols, np.random.randint(70, 90) / 100)

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/replay_attack_high.csv', index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# REPLAY ATTACK HIGH FINISHED
# # # # # # # # # # # # # ## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# zero_out_same_intervals ATTACK LOW
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'data/models_testing_before_attacks.csv'
data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)

# Attack functions

cols_of_att = ['Speed', 'Compass', 'Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'GNSS_latitude', 
                'GNSS_longitude', 'GNSS_altitude', 'Control_throttle', 'Control_steer']


# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
        random_number = np.random.randint(1, 4)

        # Select a random column(s) to attack based on the random number
        column_to_attack = random.sample(cols, random_number)

        # Apply the attack
        group = zero_out_same_intervals(group, column_to_attack)

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/zero_out_same_intervals_low.csv', index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# zero_out_same_intervals ATTACK LOW FINISHED
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# zero_out_same_intervals ATTACK MEDIUM
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'data/models_testing_before_attacks.csv'
data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)

# Attack functions

cols_of_att = ['Speed', 'Compass', 'Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'GNSS_latitude', 
                'GNSS_longitude', 'GNSS_altitude', 'Control_throttle', 'Control_steer']


# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
        random_number = np.random.randint(4, 7)

        # Select a random column(s) to attack based on the random number
        column_to_attack = random.sample(cols, random_number)

        # Apply the attack
        group = zero_out_same_intervals(group, column_to_attack)

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/zero_out_same_intervals_medium.csv', index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# zero_out_same_intervals ATTACK MEDIUM FINISHED
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# zero_out_same_intervals ATTACK HIGH
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load data
data_path_1 = 'data/models_testing_before_attacks.csv'
data1 = pd.read_csv(data_path_1)

# Create a label column
data1["Label"] = 0  # initialize all as normal

# CREATE ATTACKS
# Number of groups
num_groups = len(data1) // 25

# Generate a list of group indexes that will be attacked
attacked_group_indexes = np.random.choice(num_groups, size=num_groups // 2, replace=False)

# Attack functions

cols_of_att = ['Speed', 'Compass', 'Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'GNSS_latitude', 
                'GNSS_longitude', 'GNSS_altitude', 'Control_throttle', 'Control_steer']


# Create groups and apply attacks/classification
groups = []
for i in range(num_groups):
    group = data1.iloc[i*25 : (i+1)*25]
    
    if i in attacked_group_indexes:
        random_number = np.random.randint(7, 11)

        # Select a random column(s) to attack based on the random number
        column_to_attack = random.sample(cols, random_number)

        # Apply the attack
        group = zero_out_same_intervals(group, column_to_attack)

        # Label the group as attacked
        group["Label"] = 1

    else:

        # Label the group as normal
        group["Label"] = 0

    groups.append(group)

all_data = pd.concat(groups)
all_data.to_csv('data/attacks_data/zero_out_same_intervals_high.csv', index=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# zero_out_same_intervals ATTACK HIGH FINISHED
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
