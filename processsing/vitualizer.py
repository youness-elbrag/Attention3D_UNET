from renderiing3d import ImageToGIF ,Image3dToGIF3d , CorrectedPrceess
from IPython.display import Image as show_gif
import argparse
import nibabel as nib
import nilearn as nl
import numpy as np
import nrrd
import h5py
from tqdm import tqdm

from IPython.display import Image as show_gif

import warnings
warnings.simplefilter("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(exit_on_error=False)
parser = argparse.ArgumentParser("--help",description='tool for MRI rendering images using python Python ''Procssing dataset Brats2020 ')                                              
parser.add_argument('--v2Drender',action='store_true'
,help="rendering 2D images from nii file took all sclices ")
parser.add_argument('--v3Drender',action='store_true'
,help="rendering the 3D nii file this may take while to be finished .")
parser.add_argument('--corrected_samples',action='store_true'
,help="virtualize the corrected sample according to Orign image.")

parser.add_argument('--type_plot'
,help="virtualize the corrected sample according to Orign image.")

args = parser.parse_args()

import warnings
warnings.simplefilter("ignore")
#few sample of daraset Brasts cancer
sample_filename = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1.nii'
sample_filename_mask = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii'
sample_corrected='BraTS2020_TrainingDataCorrected/n4bais/MICCAI_BraTS2020_TrainingData/BraTS20_Training_126/BraTS20_Training_126_t1.nii'
save_path_img='images_processing/'
sample_img = nib.load(sample_filename)
sample_img = np.asanyarray(sample_img.dataobj)
sample_mask = nib.load(sample_filename_mask)
sample_mask = np.asanyarray(sample_mask.dataobj)

print("img shape ->", sample_img.shape)
print("mask shape ->", sample_mask.shape)

## matching colormaps
#Greys_r RdGy_r  CMRmap afmhot binary_r bone copper cubehelix gist_heat gist_stern gnuplot hot inferno magma nipy_spectral
try:
    if args.v2Drender :
        sample_data_gif = ImageToGIF()
        label = sample_filename.replace('/', '.').split('.')[-2]
        filename = save_path_img+label+'_2d.gif'

        for i in range(sample_img.shape[0]):
            image = np.rot90(sample_img[i])
            mask = np.clip(np.rot90(sample_mask[i]), 0, 1)
            sample_data_gif.add(image, mask, label=f'{label}_{str(i)}')
        
        sample_data_gif.save(filename, fps=15)
        show_gif(filename, format='png')

    elif args.v3Drender:

            title = sample_filename.replace(".", "/").split("/")[-2]
            print(title)
            filename = save_path_img+title+"younes_3d.gif"
            #img_dim = (120, 120, 78)
            data_to_3dgif = Image3dToGIF3d()#
            transformed_data = data_to_3dgif.get_transformed_data(sample_img)
            #print(transformed_data)
            data_to_3dgif.plot_cube(
                transformed_data[:38, :47, :35],#[:77, :105, :55]
                title=title,
                make_gif=True,
                path_to_save=filename
            )
            show_gif(filename, format='png')

    elif args.corrected_samples:
           #the function to plot the corrected with oring img 
            #     type_virtualizer{
            #     option 1 = Anat ,
            #     option 2 = epi ,
            #     option 3 = img ,
            #     
            #  }
            output=CorrectedPrceess(sample_filename,sample_corrected) 
            output.virtualize_bias(save_path_img,args.type_plot) 
            #print(output) 
    else  : #print("may this could happned because of :"+err)    
         ValueError("segmentation dume core \n, check if you laptop support GPU")
    
except:
        Exception("something goes wrong ,,, please check the path or file")       

