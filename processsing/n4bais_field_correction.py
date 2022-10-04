import shutil
import SimpleITK as sitk
import glob
import os
from nipype.interfaces.ants import N4BiasFieldCorrection

modalities = ["flair", "t1", "t1ce", "t2"]
input_file_data='/content/BraTS2020_TrainingData'
output_file_data_corrected='/content/BraTS2020_TrainingData_correted_N4Bias'

def correct_bias(in_path, out_path, image_type=sitk.sitkFloat64):
    # N. Tustison et al., N4ITK: Improved N3 Bias Correction, IEEE Transactions on Medical Imaging, 29(6):1310-1320, June 2010.
  
      input_image = ants.image_read(in_path)
      ants.n4_bias_field_correction(input_image).to_file(out_path)
      #sitk.WriteImage(output_image, out_path)
      return os.path.abspath(out_path)

def get_image_path(subject_folder, name):
    file_name = os.path.join(subject_folder, "*" + name + ".nii")
    return glob.glob(file_name)[0]

def check_origin(in_path, in_path2):
    image = sitk.ReadImage(in_path)
    image2 = sitk.ReadImage(in_path2)
    if not image.GetOrigin() == image2.GetOrigin():
        image.SetOrigin(image2.GetOrigin())
        sitk.WriteImage(image, in_path)

def normalize_image(in_path, out_path, bias_correction=True):
    if bias_correction:
        correct_bias(in_path, out_path)
    else:
        shutil.copy(in_path, out_path)

def preprocess_brats_folder(in_folder, out_folder, truth_name='seg', no_bias_correction_modalities=None):
    for name in modalities:
        image_image = get_image_path(in_folder, name)
        case_ID = os.path.basename(out_folder)
        out_path = os.path.abspath(os.path.join(out_folder, "%s_%s.nii"%(case_ID, name)))
        perform_bias_correction = no_bias_correction_modalities and name not in no_bias_correction_modalities
        normalize_image(image_image, out_path, bias_correction=perform_bias_correction)

    truth_image = get_image_path(in_folder, truth_name)
    out_path = os.path.abspath(os.path.join(out_folder, "%s_truth.nii"%(case_ID)))
    shutil.copy(truth_image, out_path)
    check_origin(out_path, get_image_path(in_folder, modalities[0])) # check with the flair image

def preprocess_brats_data(brats_folder, out_folder, overwrite=False, no_bias_correction_modalities=("flair")):
    for subject_folder in glob.glob(os.path.join(brats_folder, "*", "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder, os.path.basename(os.path.dirname(subject_folder)),
                                              subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    os.makedirs(new_subject_folder)
                preprocess_brats_folder(subject_folder, new_subject_folder,
                                     no_bias_correction_modalities=no_bias_correction_modalities)

def main(brats_path, preprocessed_brats):
    preprocess_brats_data(brats_path, preprocessed_brats)

if __name__ == "__main__":
    main(brats_path=input_file_data, preprocessed_brats=output_file_data_corrected)