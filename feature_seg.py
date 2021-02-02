# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:49:37 2020

@author: Zhiyun Gong
"""

from __future__ import print_function
import six
# import os  # needed navigate the system to get the input data
import pandas as pd
import numpy as np
from radiomics import featureextractor  # This module is used for interaction with pyradiomics
from pathlib import Path
from shutil import copyfile
import glob
import os
import SimpleITK as sitk

def collect_modality(modal):
    files = []
    for file in Path('Pre-operative_TCGA_GBM_NIfTI_and_Segmentations').glob('**/*' + modal +'.nii.gz'):
        files.append(file)
    return files


t1_Path = 'features/t1/'
t1Gd_Path = 'features/t1Gd/' 
t2_Path = 'features/t2/'
flair_Path = 'features/flair/'

# Read patient ids
patient_ids = pd.read_csv('features/patient_ids.txt', header=None)[0]

# Segmentation results
yuanqi_files = glob.glob('features/yuanqi_seg/*.nii.gz')

# Instantiate the extractor
global geometryTolerance

geometryTolerance = 100000
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor._setTolerance()

def extract_radio_ft(patient_ids, modal_path, seg_path):
    features = []
    for p in patient_ids:
        seg_file = seg_path + p + '.nii.gz'
        img_file = glob.glob(os.path.join(modal_path, p+'*'))[0]
        result = extractor.execute(img_file, seg_file)
        curr_res=[]
        for key, value in six.iteritems(result):
            if key == 'diagnostics_Mask-original_CenterOfMassIndex':
                for i in range(3):
                    curr_res.append(value[i])
                    # feature_names.append(key + "_" + str(i))
            if type(value) == np.ndarray or type(value) == np.float64 or type(value) == int:                
                if type(value) == np.ndarray:
                    value = value.reshape(-1,1)[0,0]
                curr_res.append(value)

        features.append(curr_res)
    return features

yuanqi_t1 = extract_radio_ft(patient_ids,t1_Path, 'features/yuanqi_seg/')
yuanqi_t1Gd = extract_radio_ft(patient_ids,t1Gd_Path, 'features/yuanqi_seg/')
yuanqi_t2 = extract_radio_ft(patient_ids,t2_Path, 'features/yuanqi_seg/')
yuanqi_flair = extract_radio_ft(patient_ids,flair_Path, 'features/yuanqi_seg/')

t1_ft_df = pd.DataFrame(yuanqi_t1)
t1_ft_df.to_csv('features/yuanqi_features/t1_radio_features.csv', index=False, header=False)  

t1Gd_ft_df = pd.DataFrame(yuanqi_t1Gd)
t1Gd_ft_df.to_csv('features/yuanqi_features/t1Gd_radio_features.csv', index=False, header=False)  

t2_ft_df = pd.DataFrame(yuanqi_t2)
t2_ft_df.to_csv('features/yuanqi_features/t2_radio_features.csv', index=False, header=False)  

flair_ft_df = pd.DataFrame(yuanqi_flair)
flair_ft_df.to_csv('features/yuanqi_features/flair_radio_features.csv', index=False, header=False)  

#--------------------- Yuying's ------------------------------
yuying_t1 = extract_radio_ft(patient_ids,t1_Path, 'features/yuying_seg/')
yuying_t1Gd = extract_radio_ft(patient_ids,t1Gd_Path, 'features/yuying_seg/')
yuying_t2 = extract_radio_ft(patient_ids,t2_Path, 'features/yuying_seg/')
yuying_flair = extract_radio_ft(patient_ids,flair_Path, 'features/yuying_seg/')

t1_ft_df = pd.DataFrame(yuying_t1)
t1_ft_df.to_csv('features/yuying_features/t1_radio_features.csv', index=False, header=False)  

t1Gd_ft_df = pd.DataFrame(yuying_t1Gd)
t1Gd_ft_df.to_csv('features/yuying_features/t1Gd_radio_features.csv', index=False, header=False)  

t2_ft_df = pd.DataFrame(yuying_t2)
t2_ft_df.to_csv('features/yuying_features/t2_radio_features.csv', index=False, header=False)  

flair_ft_df = pd.DataFrame(yuying_flair)
flair_ft_df.to_csv('features/yuying_features/flair_radio_features.csv', index=False, header=False)  