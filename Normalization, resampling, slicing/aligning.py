"""
Created on Fri Apr  7 11:32:41 2023

@author: yazdianp
"""

import SimpleITK as sitk
import shutil
import os


def load_nifti_image(file_path):
    return sitk.ReadImage(file_path)

def save_nifti_image(image, file_path):
    sitk.WriteImage(image, file_path)

def align_mask_to_image(image, mask):
    # Use the image's origin, spacing, and direction to align the mask
    mask.SetOrigin(image.GetOrigin())
    mask.SetSpacing(image.GetSpacing())
    mask.SetDirection(image.GetDirection())

    # Resample the mask using the image's information
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    aligned_mask = resampler.Execute(mask)

    return aligned_mask
def copy_mask_label(mask_label_path, output_label_path):
    shutil.copy(mask_label_path, output_label_path)
def main():
    image_path = r'/data/AMPrj/AllImages/NiftiEverything/Images/0040-Subject-00967/0040-18923/normalized_aligned.nii.gz'
    mask_path = r'/data/AMPrj/NiftiCombinedNew/Masks/0040-Subject-00967/0040-18923/mask.nii.gz'
    mask_label_path = '/data/AMPrj/NiftiCombinedNew/Masks/0040-Subject-00967/0040-18923/mask.txt'
    output_label_path = '/data/AMPrj/NiftiCombinedNew/Masks/0040-Subject-00967/0040-18923/mask_aligned.txt'
    output_path = r'/data/AMPrj/NiftiCombinedNew/Masks/0040-Subject-00967/0040-18923/mask_aligned.nii.gz'
    print ('loading image')
    image = load_nifti_image(image_path)
    print ('loading Mask')
    mask = load_nifti_image(mask_path)

    aligned_mask = align_mask_to_image(image, mask)
    copy_mask_label(mask_label_path, output_label_path)
    save_nifti_image(aligned_mask, output_path)
    print ('done')
    
    

if __name__ == '__main__':
    main()

