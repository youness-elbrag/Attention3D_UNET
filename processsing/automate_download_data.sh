#!/bin/bash 

echo "creating the folder ~/kaggale for testing the APi "
#after installing kaggale APi packages automatically will be created 
#folder into local directroy 
path_api="/home/yunus/.kaggle"
#creat APi kaggale account from kaggale 
api="kaggle.json"
#run commaned to download the datatset from kaggale 
cmd="kaggle datasets download -d awsaf49/brats20-dataset-training-validation"

mkcd()
{
    #make file executable 
    chmod 600 $api
    #move file into locale directory 
    mv $api $path_api
    $cmd
    #unzip the file in project directory 
    unzip brats20-dataset-training-validation.zip 
}
mkcd
echo "...finished "