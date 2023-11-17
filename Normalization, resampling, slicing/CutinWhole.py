# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:27:24 2022

@author: yazdianp
"""
import SimpleITK as sitk
import numpy as np
import os
import numpy as np
import SimpleITK as sitk
import os

def resample_img(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
	int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkBSpline)
    else:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

    return resample.Execute(itk_image)


with open('all.txt', mode="rt", newline="") as f:
  cases = [ case.strip() for case in f if len(case.strip()) > 0 ]
  print (cases)
MainRoot = 'Images'   
Cut_root = '/gpfs/gsfs10/users/AMPrj/NiftiCombinedNewCut/Images'
dst_root = '/data/AMPrj/Yazdianp/YOLOCutVenDTumor/Images'
#volumePathName = "mask_aligned .nii.gz"
for case in cases: 
    for i in range(1):
        volumePathName = "normalized4_aligned_affine.nii.gz"

        if not os.path.exists(os.path.join(Cut_root ,case)):
           os.makedirs((os.path.join(Cut_root,case)))
        else:
           ('cut folder available')
        if not os.path.exists(os.path.join(MainRoot, case, volumePathName)):
           print (f'{case}: No Volume')
           with open((os.path.join(f'ErrorsForNathan.txt')), "a+") as f:
                f.seek(0)
                f.write("\n")
                f.write(case)                       
           continue
        else: 
           Image = sitk.ReadImage(os.path.join(MainRoot, case, volumePathName))
        if not os.path.exists(os.path.join(MainRoot, case)): 
            print (f'{case}: patient not available')
            continue
        mask_path = os.path.join('Masks', case, "mask_aligned.nii.gz")
        if not os.path.exists(mask_path):
            print(f'Mask file not found: {mask_path}')
            with open((os.path.join(f'Mask file not found.txt')), "a+") as f:
                f.seek(0)
                f.write("\n")
                f.write(mask_path)   
            continue
        else:
            Mask = sitk.ReadImage(mask_path)  # Don't touch it is the MAIN Mask
        npImage = sitk.GetArrayViewFromImage(Image)
        npMask = sitk.GetArrayViewFromImage(Mask)
        halfX = npImage.shape[2]//2       
        npMaskRight = npMask[..., :halfX]
        npMaskLeft = npMask[..., halfX:]
        labeledVolumeRight = (npMaskRight > 0).sum()
        labeledVolumeLeft = (npMaskLeft > 0).sum()

        if labeledVolumeLeft > 0.01*labeledVolumeRight:
            print (f"{case}: npVolumeLeft")
            npVolumeLeft = npImage[..., halfX:]
            npFakeRightImage = npVolumeLeft[...,::-1]
            _npVolumeRight = npMask[..., :halfX]
            #WholeVolume = npVolumeLeft #CutTest
            WholeVolume = np.concatenate((npFakeRightImage, npVolumeLeft),axis=2) #Image
            #WholeVolume = np.concatenate((_npVolumeRight,npVolumeLeft),axis=2) #MASK
            #WholeVolume = np.concatenate((npFakeRightImage,_npVolumeRight),axis=2) 
            theSlice = sitk.GetImageFromArray(WholeVolume , isVector=False)
            if not os.path.isdir((os.path.join(Cut_root ,case))):
                os.makedirs((os.path.join(Cut_root,case)))
            sitk.WriteImage(theSlice, (os.path.join(Cut_root,case,volumePathName)))  

        if labeledVolumeRight > 0.01*labeledVolumeLeft:
            print (f"{case}: npVolumeRight")
            npVolumeRight = npImage[..., :halfX]
            npFakeLeftImage = npVolumeRight[...,::-1 ]
            _npVolumeLeft = npMask[..., halfX:]
            #WholeVolume = npVolumeRight #CutTest
            WholeVolume = np.concatenate((npVolumeRight,npFakeLeftImage),axis=2) 
            #WholeVolume = np.concatenate((npVolumeRight,_npVolumeLeft),axis=2) #MASK
            #WholeVolume=np.concatenate((_npVolumeLeft,npFakeLeftImage),axis=2)
            theSlice = sitk.GetImageFromArray(WholeVolume , isVector=False)
            if not os.path.isdir((os.path.join(Cut_root ,case))):
                os.makedirs((os.path.join(Cut_root,case)))
            sitk.WriteImage(theSlice, (os.path.join(Cut_root,case,volumePathName)))     


    channels = 1
    volumes = []
    for i in range(channels):
        volumePathName = "normalized4_aligned_affine.nii.gz"
        print (volumePathName)
        if not os.path.isdir((os.path.join(dst_root ,case))):#change
            os.makedirs((os.path.join(dst_root,case)))#change

        if not os.path.isdir((os.path.join(Cut_root,case))):#change
            continue

        image = os.path.join(Cut_root, case, volumePathName)#change
        if os.path.exists(image):
            itk_Image = sitk.ReadImage(image)
        else:
          print(f"Image file not found: {image}")
          with open((os.path.join(f'Image file not found.txt')), "a+") as f:
            f.seek(0)
            f.write("\n")
            f.write(image)   
          continue

        itk_Image = sitk.ReadImage(image)
        resampled_sitk_img = resample_img(itk_Image, out_spacing=[1.0, 1.0, 1.0], is_label=False)
        volumes.append(resampled_sitk_img)
    if len(volumes) == 0:
        print("No volumes found for this case. Skipping.")
        continue

    npVolumes = [ sitk.GetArrayViewFromImage(volume) for volume in volumes ]
    assert all ((npVolume.shape == npVolumes[0].shape for npVolume in npVolumes))
    npSlice = np.zeros([npVolumes[0].shape[1], npVolumes[0].shape[2], channels], dtype=npVolumes[0].dtype)
    # Numpy convention is Z, Y, X
    npSlice = np.zeros([npVolumes[0].shape[1], npVolumes[0].shape[2], channels], dtype=npVolumes[0].dtype)
    for z in range(npVolumes[0].shape[0]):
        for c in range(channels):
            npSlice[..., c] = npVolumes[c][z, ...]
        fname = volumePathName .replace('_affine.nii.gz',f"{z+1}")#change
        # "slice" is a built-in function
        theSlice = sitk.GetImageFromArray(npSlice, isVector=True)
        theSlice.SetSpacing([volumes[0].GetSpacing()[1], volumes[0].GetSpacing()[2]])
        if not os.path.exists((os.path.join(dst_root ,case))):#change
            os.makedirs((os.path.join(dst_root,case)))#change
        img_save_path = os.path.join (dst_root,case)  	
        img_f_path = os.path.join( img_save_path, fname + ".nii.gz")
        sitk.WriteImage(theSlice, img_f_path) # May need to change the order of arguments
