# Attention_3DUnetPP_Brain-Tumor-Segementation
this repo contain the implementaiion of paper we are working on wih the team from Jorden Univesity  of science and technology , in this work we present a new based model on Neasted 3DUnet++ combined wiht attention mechanism Block for more effiency feature extracttion 

## the problem case study in 3D Biomedical image processing

- subject
Accurate segmentation of brain tumor sub-regions is essential in the quantifica- tion of lesion burden, providing insight into the functional outcome of patients. In this regard, 3D multi-parametric magnetic resonance imaging (3D mpMRI) is widely used for non-invasive visualization and analysis of brain tumors. Different MRI sequences (such as T1, T1ce, T2, and FLAIR) are often used to provide complementary information about different brain tumor sub-regions

- Problem
in many cases for processing Medical images to get better understanding of disease and impact on human being life such Brain tumor is most area for reseachers to improve system diagnosis in partuclar Task Segementation , last few year lunch of challenge BRATS for segmentation Brain tumor Sub-regrion many of studying came up to improve CAD system ,

- Solution
For automatic segmentation we will use Unet3d To predict the age and number of days of survival: first, we will train the auto-encoder to scale the space from 4 240 240 * 150 to 512, and then extract the statistical values, ​​and hidden representations for each identifier in the data encoded by the pre-trained auto-encoder and based on this tabular data we will train SVR

1. [introduction](#introduction)
2. [environment_project](#environment_project)
3. [run_project]
5. [model]
5. [Results]

### introduction
*Imaging Data Description*

All BraTS multimodal scans are available as NIfTI files (.nii.gz) and describe a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes, and were acquired with different clinical protocols and various scanners from multiple (n=19) institutions, mentioned as data contributors here.

All the imaging datasets have been segmented manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1), as described both in the BraTS 2012-2013 TMI paper and in the latest BraTS summarizing paper. The provided data are distributed after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same resolution (1 mm^3) and skull-stripped.

### environment_project
* setup the enviroment;
	* install script shell ;
       here you will need to run script shell to install all the dependencies needed for 
       run code :
       1. create [kaggle](https://www.kaggle.com/) account to access to the data API 
       2. add path kaggle.json to script shell $path_api
       3. create the enviromenet here you will need to run 
       
                python create_env.py {name of your env}

       4. make sure the requirements.txt exist to the repo 
       * install the packges if you want fisrt neeed to run 

                pip install -r requirements.txt 

                chmod +x automate_downlaod_data.sh && ./automate_downlaod_data.sh

* PerProcessig dataset Brast2020;

    this tool built based on top of BET algorithm that publish from [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET) and [N4baisCorrection](https://pubmed.ncbi.nlm.nih.gov/20378467/) we automated the process and handle the data in 3D shape

	* tool description;

        we develpoed a simple tool that helps to Post Processing the dstaset 
        1. N4 bais Correction field this will increase the Low intensity of the image to run :
               python Postprocessing --path {path_name} --n4baiscorrection

        2. Skull Stripping this technic helps to reduce tissues such skull and midbrain .. only we do care about in our project is brain tissues to tun it :

                     python Postprocessing --path {path_name} --skull_stripping 

	* Item 2.2;
* Item 3
	* Item 3.1;
		* Item 3.1.1;
D 