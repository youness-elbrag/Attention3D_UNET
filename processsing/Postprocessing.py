import argparse
from nipype.interfaces.ants import N4BiasFieldCorrection
from scipy import ndimage as nd
import ants

import nipype.interfaces.fsl as fsl
from nipype.interfaces.fsl import BET, IsotropicSmooth
from Posthelper import *
from pathlib import Path
input_file_data='BraTS2020_TrainingData'
#output_file_data_corrected_n4bais='/content/BraTS2020_TrainingData_correted_N4Bias'

error_list = []
Normlize_VS = [1, 1, 1]

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser("--help",description='tool for MRI artifics using python Python '
                                                    'Procssing dataset Brats2020 ')
parser.add_argument('--path', metavar='ENV_DIR', nargs='+',
                        help='A directory to create the environment in.')                                               
parser.add_argument('--n4baisfieldcorrection',action='store_true',
help="A directory to save corrected samples n4baisfieldcorrection ")
parser.add_argument('--skull_stripping',action='store_true',
help="A directory to save corrected samples Skull stripping.")

args = parser.parse_args() 

def n4baisfieldcorrection(in_path, out_path, image_type=sitk.sitkFloat64):
    # N. Tustison et al., N4ITK: Improved N3 Bias Correction, IEEE Transactions on Medical Imaging, 29(6):1310-1320, June 2010.
  
      input_image = ants.image_read(in_path)
      ants.n4_bias_field_correction(input_image).to_file(out_path)
      #sitk.WriteImage(output_image, out_path)
      return os.path.abspath(out_path)

def skull_stripping(in_path, out_path, image_type=sitk.sitkFloat64):
    # N. Tustison et al., N4ITK: Improved N3 Bias Correction, IEEE Transactions on Medical Imaging, 29(6):1310-1320, June 2010.
    try:
        BET(in_file=in_path,
            out_file = out_path).run()
        
    except:
        error_list.append(out_path)
        print(error_list)

        pass
      #sitk.WriteImage(output_image, out_path)
    return os.path.abspath(out_path)  
        
def normalize_image(in_path, out_path, correction=True):
    if correction and args.n4baisfieldcorrection:
        n4baisfieldcorrection(in_path, out_path)
    else:
        shutil.copy(in_path, out_path)
    
    if correction and args.skull_stripping:
        skull_stripping(in_path, out_path)  
    else:
        shutil.copy(in_path, out_path)

def main(brats_path, preprocessed_brats):
    preprocess_brats_data(brats_path, preprocessed_brats)

if __name__ == "__main__":                   
    if args.path and args.n4baisfieldcorrection:
        main(brats_path=input_file_data, preprocessed_brats=str(args.path))
    elif args.path and args.skull_stripping:
              main(brats_path=input_file_data, preprocessed_brats=str(args.path))
    else:
        print("please inster the full agrument CLI tool") 
    


