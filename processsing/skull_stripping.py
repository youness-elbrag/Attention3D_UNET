import shutil
import SimpleITK as sitk
import glob
import os
import nibabel as nib
# import the BrainExtractor class
from brainextractor import BrainExtractor
from scipy import ndimage as nd
#from deepbrain import Extractor
# import nipype.interfaces.fsl as fsl
from nipype.interfaces.fsl import BET, IsotropicSmooth
from nipype.interfaces.fsl import Info

Info.version()  

print(Info.output_type())

modalities = ["flair", "t1", "t1ce", "t2"]
# input_file_data='/content/BraTS2020_TrainingData_correted_N4Bias'
# output_file_data_corrected='/content/BraTS2020_TrainingData_Skull_Stripping_and_N4'
input_file_data='BraTS2020_TrainingData'
output_file_data_corrected='/content/BraTS2020_TrainingData_correted_Skull'


error_list = []
# Normlize_VS = [1, 1, 1]

# def ThreeD_resize(imgs, Normlize_VS, img_vs): 
#    #order The order of the spline interpolation, default is 3. The order has to be in the range 0-5. 
#    zf0 = img_vs[0] / Normlize_VS[0] 
#    zf1 = img_vs[1] / Normlize_VS[1] 
#    zf2 = img_vs[2] / Normlize_VS[2] 
#    new_imgs = nd.zoom(imgs, [zf0, zf1, zf2], order=0) 
#    return new_imgs


# def Skull_Stripping(in_path, out_path):
#         data_img = sitk.ReadImage(in_path)
#         array_img = sitk.GetArrayFromImage(data_img)
#         Spacing_img = data_img.GetSpacing()[::-1]
#         Img_array = ThreeD_resize(array_img, Normlize_VS, Spacing_img)
#         Img_ni = sitk.GetImageFromArray(Img_array)
#         Img_ni.SetDirection(data_img.GetDirection())
#         sitk.WriteImage(Img_ni,out_path)
#     # N. Tustison et al., N4ITK: Improved N3 Bias Correction, IEEE Transactions on Medical Imaging, 29(6):1310-1320, June 2010.
       
#         return os.path.abspath(out_path)

# def get_image_path(subject_folder, name):
#     file_name = os.path.join(subject_folder, "*" + name + ".nii")
#     return glob.glob(file_name)[0]

# def check_origin(in_path, in_path2):
#     image = sitk.ReadImage(in_path)
#     image2 = sitk.ReadImage(in_path2)
#     if not image.GetOrigin() == image2.GetOrigin():
#         image.SetOrigin(image2.GetOrigin())
#         sitk.WriteImage(image, in_path)

# def reduce_Skull_image(in_path, out_path,skull_Stripping=True):
#     if skull_Stripping:
#         Skull_Stripping(in_path, out_path)
#     else:
#         shutil.copy(in_path, out_path)



# def preprocess_brats_folder(in_folder, out_folder, truth_name='seg', no_skull_Stripping_modalities=None):
#     for name in modalities:
#         image_image = get_image_path(in_folder, name)
#         case_ID = os.path.basename(out_folder)
#         out_path = os.path.abspath(os.path.join(out_folder, "%s_%s.nii"%(case_ID, name)))
#         perform_bias_correction = no_skull_Stripping_modalities and name not in no_skull_Stripping_modalities
#         reduce_Skull_image(image_image, out_path)

#     truth_image = get_image_path(in_folder, truth_name)
#     out_path = os.path.abspath(os.path.join(out_folder, "%s_sge.nii"%(case_ID)))
#     shutil.copy(truth_image, out_path)
#     check_origin(out_path, get_image_path(in_folder, modalities[0])) # check with the flair image

# def preprocess_brats_data(brats_folder, out_folder, overwrite=False, no_skull_Stripping_modalities=("flair")):
#     for subject_folder in glob.glob(os.path.join(brats_folder, "*", "*")):
#         if os.path.isdir(subject_folder):
#             subject = os.path.basename(subject_folder)
#             new_subject_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)),
#                                               subject)
#             if not os.path.exists(new_subject_folder) or overwrite:
#                 if not os.path.exists(new_subject_folder):
#                     os.makedirs(new_subject_folder)
#                 preprocess_brats_folder(subject_folder, new_subject_folder,
#                                      no_skull_Stripping_modalities=no_skull_Stripping_modalities)

# def main(brats_path, preprocessed_brats):
#     preprocess_brats_data(brats_path, preprocessed_brats)

# if __name__ == "__main__":
#     main(brats_path=input_file_data, preprocessed_brats=output_file_data_corrected)
    