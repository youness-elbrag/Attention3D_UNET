# Attention_3DUnetPP_Brain-Tumor-Segementation
this repo contain the implementaiion of paper we are working on wih the team from Jorden Univesity  of science and technology , in this work we present a new based model on Neasted 3DUnet++ combined wiht attention mechanism Block for more effiency feature extracttion 

## the problem case study in 3D Biomedical image processing

- subject
Accurate segmentation of brain tumor sub-regions is essential in the quantifica- tion of lesion burden, providing insight into the functional outcome of patients. In this regard, 3D multi-parametric magnetic resonance imaging (3D mpMRI) is widely used for non-invasive visualization and analysis of brain tumors. Different MRI sequences (such as T1, T1ce, T2, and FLAIR) are often used to provide complementary information about different brain tumor sub-regions

- Problem
in many cases for processing Medical images to get better understanding of disease and impact on human being life such Brain tumor is most area for reseachers to improve system diagnosis in partuclar Task Segementation , last few year lunch of challenge BRATS for segmentation Brain tumor Sub-regrion many of studying came up to improve CAD system ,

- Solution
For automatic segmentation we will use Unet3d To predict the age and number of days of survival: first, we will train the auto-encoder to scale the space from 4 240 240 * 150 to 512, and then extract the statistical values, ​​and hidden representations for each identifier in the data encoded by the pre-trained auto-encoder and based on this tabular data we will train SVR

1. [introduction](introduction)
2. [environment_project]
3. [run_project]
5. [model]
5. [Results]

###### introcction
*Imaging Data Description*

All BraTS multimodal scans are available as NIfTI files (.nii.gz) and describe a) native (T1) and b) post-contrast T1-weighted (T1Gd), c) T2-weighted (T2), and d) T2 Fluid Attenuated Inversion Recovery (T2-FLAIR) volumes, and were acquired with different clinical protocols and various scanners from multiple (n=19) institutions, mentioned as data contributors here.

All the imaging datasets have been segmented manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1), as described both in the BraTS 2012-2013 TMI paper and in the latest BraTS summarizing paper. The provided data are distributed after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same resolution (1 mm^3) and skull-stripped.
