import os 
import os.path 

#we need to downlaod the data from the kaggle 
download_data="kaggle datasets download -d awsaf49/brats20-dataset-training-validation"

# create the API token from kaggle account 
chmod_file='chmod 600 kaggle.json'
path_file="kaggle.json"

try:
    os.path.exists(path_file)
except FileNotFoundError():
    print("you need fisrt to create API token from kaggale account ")   
else :
    os.system(chmod_file)
  
os.system(download_data)