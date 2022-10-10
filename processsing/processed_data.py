
import ants
import nipype.interfaces.fsl as fsl
from nipype.interfaces.fsl import BET 
import SimpleITK as sitk
import glob
import os
import shutil

class Normalize_Brastdata:
    error_list=[]
    def __init__(self ,in_path, out_path, correction):
    
        self.in_path=in_path
        self.out_path=out_path
        self.correction=correction

    def skull_stripping(self,image_type=sitk.sitkFloat64):
        try:
            BET(in_file=self.path,
            out_file=self.out_path).run()
        except:
               error_list.append(self.out_path)
               print(error_list)
               pass
        return os.path.abspath(self.out_path)  
        
    def n4baisfieldcorrection(self, image_type=sitk.sitkFloat64):
    # N. Tustison et al., N4ITK: Improved N3 Bias Correction, IEEE Transactions on Medical Imaging, 29(6):1310-1320, June 2010.

      input_image = ants.image_read(self.in_path)
      ants.n4_bias_field_correction(input_image).to_file(self.out_path)
      #sitk.WriteImage(output_image, out_path)
      return os.path.abspath(self.out_path)

    def __call__(self,command):  

        if self.correction == True and command=='n4baisfieldcorrection':
             n4baisfieldcorrection(self.in_path,self.out_path)

        elif self.correction == True and command =='skull_strippinng':
             skull_stripping(self.in_path, self.out_path)    
        else:
            shutil.copy(self.in_path, self.out_path)
            

