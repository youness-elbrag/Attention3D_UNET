# Attention_3DUnetPP_Brain-Tumor-Segementation
this repo contain the implementaiion of paper we are working on wih the team from Jorden Univesity  of science and technology , in this work we present a new based model on Neasted 3DUnet++ combined wiht attention mechanism Block for more effiency feature extracttion 

## the problem case study in 3D Biomedical image processing

1. subject
Accurate segmentation of brain tumor sub-regions is essential in the quantifica- tion of lesion burden, providing insight into the functional outcome of patients. In this regard, 3D multi-parametric magnetic resonance imaging (3D mpMRI) is widely used for non-invasive visualization and analysis of brain tumors. Different MRI sequences (such as T1, T1ce, T2, and FLAIR) are often used to provide complementary information about different brain tumor sub-regions

2. Problem
in many cases for processing Medical images to get better understanding of disease and impact on human being life such Brain tumor is most area for reseachers to improve system diagnosis in partuclar Task Segementation , last few year lunch of challenge BRATS for segmentation Brain tumor Sub-regrion many of studying came up to improve CAD system ,

3. Solution
For automatic segmentation we will use Unet3d To predict the age and number of days of survival: first, we will train the auto-encoder to scale the space from 4 240 240 * 150 to 512, and then extract the statistical values, ​​and hidden representations for each identifier in the data encoded by the pre-trained auto-encoder and based on this tabular data we will train SVR
