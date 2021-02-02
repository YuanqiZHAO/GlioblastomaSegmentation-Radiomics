# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 18:28:30 2020

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


def collect_modality(modal):
    files = []
    for file in Path('Pre-operative_TCGA_GBM_NIfTI_and_Segmentations').glob('**/*' + modal +'.nii.gz'):
        files.append(file)
    return files

def copy_modality(files,dest):
    for f in files:
        copyfile(f,dest+str(Path(f).name))
    return

t1_files = collect_modality('t1')
t2_files = collect_modality('t2')
t1Gd_files = collect_modality('t1Gd')
flair_files = collect_modality('flair')
mask_files = collect_modality('ManuallyCorrected')

# Segmentation results
yuanqi_files = glob.glob('features/yuanqi_seg/*.nii.gz')

    

# Copy files to separate folder
copy_modality(mask_files,'features/masks/')
copy_modality(t1_files,'features/t1/')
copy_modality(t1Gd_files,'features/t1Gd/')
copy_modality(t2_files,'features/t2/')
copy_modality(flair_files,'features/flair/')


#

# Instantiate the extractor
extractor = featureextractor.RadiomicsFeatureExtractor()

print('Extraction parameters:\n\t', extractor.settings)
print('Enabled filters:\n\t', extractor.enabledImagetypes)
print('Enabled features:\n\t', extractor.enabledFeatures)


# file_prefix = mask_files[0].name.split('_')[0] + '_' + mask_files[0].name.split('_')[1]
t1_Path = 'D:/20Fall/02-740/Project/features/t1/'
t1Gd_Path = 'D:/20Fall/02-740/Project/features/t1Gd/' 
t2_Path = 'D:/20Fall/02-740/Project/features/t2/'
flair_Path = 'D:/20Fall/02-740/Project/features/flair/'

# maskPath = 'D:/20Fall/02-740/Project/features/masks/'

def get_patient_ids(files):
    ids = []
    for f in files:
        ids.append(Path(f).name.split('_')[0])
    return ids


def get_modal_filename(label_fname, modal):
    file_prefix = label_fname.name.split('_')[0] + '_' + label_fname.name.split('_')[1]
    # file_prefix = label_fname.name.split('_')[0]
    modal_fname = file_prefix + "_" +  modal + ".nii.gz"
    return modal_fname

def get_modal_filename_new(label_fname,imagePath, modal):
    modal_fname = glob.glob(os.path.join(imagePath, label_fname.split('.')[0]+'*'))
    # modal_fname = label_fname.split('.') + '_' + modal + ".nii.gz"
    return modal_fname

def extract_feature(mask_files,modal, imagePath, maskPath):
    features = []
    feature_names = []
    for f in mask_files:
        modal_fname = imagePath + get_modal_filename(f, modal)
        # print(get_modal_filename_new(f, imagePath, modal))
        # modal_fname = imagePath + get_modal_filename_new(f, imagePath, modal)[0]

        print(modal_fname)
        # label_fname = maskPath + f.name
        label_fname = f
        result = extractor.execute(modal_fname, label_fname)
        curr_res=[]
        for key, value in six.iteritems(result):
            if key == 'diagnostics_Mask-original_CenterOfMassIndex':
                for i in range(3):
                    curr_res.append(value[i])
                    feature_names.append(key + "_" + str(i))
            if type(value) == np.ndarray or type(value) == np.float64 or type(value) == int:                
                if type(value) == np.ndarray:
                    value = value.reshape(-1,1)[0,0]
                curr_res.append(value)
                feature_names.append(key)

        features.append(curr_res)
    return features, feature_names[:115]

# Yuanqi's
# maskPath = 'features/yuanqi_seg/'

# t1_features_yz,_ = extract_feature(yuanqi_files,'t1', t1_Path, maskPath)

#-------------------------------------

# Extract features from T1 images
t1_features,feature_names = extract_feature(mask_files,'t1', t1_Path, maskPath)
t1_ft_df = pd.DataFrame(t1_features)
t1_ft_df.iloc[:3,:3]
t1_ft_df.to_csv('features/t1_radio_features.csv', index=False, header=False)  

# Extract features from T1GD images
t1Gd_features,feature_names = extract_feature(mask_files,'t1Gd', t1Gd_Path, maskPath)
t1Gd_ft_df = pd.DataFrame(t1Gd_features)
t1Gd_ft_df.iloc[:3,:10]
t1Gd_ft_df.to_csv('features/t1Gd_radio_features.csv', index=False, header=False)  


# Extract features from T2 images
t2_features,_ = extract_feature(mask_files,'t2', t2_Path, maskPath)
t2_ft_df = pd.DataFrame(t2_features)
t2_ft_df.iloc[:3,:3]
t2_ft_df.to_csv('features/t2_radio_features.csv', index=False, header=False)  


# Extract features from FLAIR images
flair_features,_ = extract_feature(mask_files,'flair', flair_Path, maskPath)
flair_ft_df = pd.DataFrame(flair_features)
flair_ft_df.iloc[:3,:3]
flair_ft_df.to_csv('features/flair_radio_features.csv', index=False, header=False)  


# Save patient ids and feature names to file
patient_ids = pd.DataFrame(get_patient_ids(mask_files))
patient_ids.to_csv('features/patient_ids.txt', index=False, header=False)
colnames = pd.DataFrame(feature_names)
colnames.to_csv('features/feature_names.txt', index=False, header=False)
t1_ft_df = pd.DataFrame(t1_features).iloc[:,22:]
t1_ft_df.to_csv('features/t1_radio_features.csv', index=False, header=False)  


    