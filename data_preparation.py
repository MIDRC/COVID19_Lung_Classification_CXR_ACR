"""
Created on Thu Aug 26 09:17:37 2021

@author: kschmidt2
"""
import os
import pydicom
import pandas as pd
import numpy as np
from PIL import Image
import random 
import logging
import numpy as np
log = logging.getLogger(__name__)




def generate_data_splits(preprocessed_files, mode):
    random.Random(23).shuffle(preprocessed_files)
    if mode=='train':
        data_len = len(preprocessed_files)
        train = preprocessed_files[0:round(0.7*data_len)]
        val = preprocessed_files[round(0.7*data_len):round(0.9*data_len)]
        test = preprocessed_files[round(0.9*data_len):]
        datalists = {'train':train, 'val': val, 'test': test}
    if mode=='inference':
        datalists = {'test': preprocessed_files}
    return datalists


def preprocess(input_dir, output_dir, scratch_dir):
    input_df = pd.read_csv(os.path.join(input_dir, 'input.csv'))
    #logging.basicConfig(filename=os.path.join(output_dir,'preprocessing.log'),level=logging.DEBUG)
    #logging.info('Begin preprocessing files')
    success, failure = 0, 0
    preprocessed_files = []
    for ind, row in input_df.iterrows():
        uid = row['studyInstanceUID']
        filepath = row['filepath']
        #label = instance[1][2]
        try:
            ds = pydicom.dcmread(filepath)
            #only accept PA or AP views
            assert ds.ViewPosition in ['PA','AP'], \
                ('Incompatible viewposition:',ds.ViewPosition,', filepath:',filepath)
            #only accept certain sopclassuids
            assert ds.SOPClassUID in ['1.2.840.10008.5.1.4.1.1.1.1', '1.2.840.10008.5.1.4.1.1.1.1.1', '1.2.840.10008.5.1.4.1.1.7', '1.2.840.10008.5.1.4.1.1.1'], \
                ('Incompatible SOPClassUID:',ds.SOPClassUID,', filepath:',filepath)
        	#only accept certain modality
            assert ds.Modality in ['DX','CR'], \
                ('Incompatible SOPClassUID:',ds.SOPClassUID,', filepath:',filepath)
            img = ds.pixel_array

            # squeeze between 0 and 1 and invert if needed
            img = (img - img.min()) / (img.max() - img.min())
            if ds.PhotometricInterpretation == 'MONOCHROME1':
                img = 1 - img

            #remove padding around image
            nonzero = np.nonzero(img)
            x1 = np.min(nonzero[0])
            x2 = np.max(nonzero[0])
            y1 = np.min(nonzero[1])
            y2 = np.max(nonzero[1])
            img = img[x1:x2, y1:y2]

            filename = filepath.replace('/', '-')  
            full_path = os.path.join(scratch_dir, filename[:-4] + '.npy')
            np.save(full_path, img)

            success += 1
            #preprocessed_files.append({'image': full_path, 'label': label, 'studyuid': uid})
            preprocessed_files.append({'image': full_path, 'studyuid': uid})
            logging.info('Preprocessed file: %s', filepath)
        except Exception as e:
            logging.error('Failed to preprocess file: %s', filepath)
            logging.error("%s",e)
            failure +=1
    logging.info('Successful image count: %s Failed image count: %s', success, failure)
    return preprocessed_files
                    
