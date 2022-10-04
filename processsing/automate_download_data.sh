#!/bin/bash 

echo "creating the folder ~/kaggale for testing the APi "
#after installing kaggale APi packages automatically will be created 
#folder into local directroy 
#/home/<username>/.kaggle
path_api="/home/yunus/.kaggle"
#creat APi kaggale account from kaggale 
api="kaggle.json"
#run commaned to download the datatset from kaggale 
cmd="kaggle datasets download -d awsaf49/brats20-dataset-training-validation"

mkcd()
{
    #make file executable 
    if [-f kaggle.json];then
         chmod 600 $api
    else
       echo "Ops the file kaggale does not exist ${/n}make you have API kaggale account"
    fi    
    #move file into locale directory 
    mv $api $path_api
    $cmd
    #unzip the file in project directory 
    unzip brats20-dataset-training-validation.zip 
    
}
mkcd

echo ".... Stage 2 setup the Biomedical packges"

mkcd()
{
    cd ..
    if [-e requirements.txt]
    then pip install -r requirements.txt
    else echo "something goes wrong make sure the file executable"
    cd processing 
    echo " make this take well to finish be patient"
    python fslinstaller.py
    echo "-----------finished ------------------"
    echo "-----------stage 3 setup the variable environment"
<<'COMMENTS'
# Change the value for FSLDIR if you have 
# installed FSL into a different location
FSLDIR=/usr/local/fsl
. ${FSLDIR}/etc/fslconf/fsl.sh
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH

# My custom values for FSL environment variables
export FSLOUTPUTTYPE=NIFTI
COMMENTS

    echo $FSLDIR
    echo "done"
}
mkcd
