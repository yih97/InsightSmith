
import os

# Get the directory path of the current file (__init__.py)
current_dir = os.path.dirname(__file__)

# Construct the relative path to the 'model' directory
home_directory = os.path.abspath(os.path.join(__file__ ,"../.."))
model_zoo = os.path.join(home_directory, 'model_zoo')
