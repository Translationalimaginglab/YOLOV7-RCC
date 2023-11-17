# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 22:47:39 2022

@author: yazdianp
"""        
import SimpleITK as sitk
import numpy as np
import os
import numpy as np
import SimpleITK as sitk
import os
import shutil
with open('all.txt', mode="rt", newline="") as f:
  cases = [ case.strip() for case in f if len(case.strip()) > 0 ]
 
for case in cases: 
    MainRoot = os.path.join('/data/AMPrj/AllImages/NiftiEverything/Images', case) 
    Dst_root = os.path.join('/data/AMPrj/NiftiCombinedNew/Images', case)
    if os.path.exists(Dst_root): 
        print ('path exists')
        continue
    if os.path.exists(MainRoot): 
       shutil.copytree(MainRoot, Dst_root)
       print (MainRoot)
    else: 
       print ('No Segmentation')