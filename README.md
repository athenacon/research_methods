# research_methods_mini_project
CS4040: Research Methods - mini project
### Project Setup Instructions

### To run the files you need:
- Python 3.11.4
- pip  
## Creating a Virtual Environment
### To create a virtual environment, run the following command:

### For Windows:
python -m venv vir_env
### For Unix/MacOS:
python3 -m venv vir_env

## Activating the Virtual Environment
### For Windows:
vir_env\Scripts\activate
### For Unix/MacOS:
source vir_env/bin/activate

## Once the virtual environment is activated, install the required packages using pip:
pip install -r requirements.txt

## Run the files using:
python "name_of_the_file.py"

## To get the results of the models (NH1) run both model_1.py and model_2.py -> this will give u the first confusion matrix metrics, and all results for all attack types per each intensity level.
## Then inspect the Excel file to obtain the graphs (attack types and intensities graphs)  
## For the ROC and PR curves run the roc_curves.py file.
## For the chi-square test (NH2), contingency table and mosaic plot, run the chi_square_test.py
## CARLA SIMULATION FRAMEWORK
## To inspect this, check carla_simulation_framework.py file. You would need to have installed the CARLA simulator which can take a significant amount of time to download. Make sure to install the version corresponding to the file.
## Attacks are implemented in the attacks directory
## All datasets + results of the datasets can be found in the data directory.
