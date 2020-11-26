from __future__ import print_function
import glob
import pandas as pd
import re

import os
import logging

import numpy as np
import time

def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging


def write_csv(file_pattern):
    #file_pattern = './data/CASIA/*/*'
    files =glob.glob(file_pattern)
    # if you want sort files according to the digits included in the filename, you can do as following:
    files = sorted(files, key=lambda x:float(re.findall("(\d+)",x)[0]))
    img_path = []
    img_labels = []
    for file_ in files:
        img_path.append(file_)
        img_labels.append(file_.split('/')[3])                                                      
    df = pd.DataFrame(list(zip(img_path, img_labels)), columns = ['image','subject'])
    df.subject = pd.Categorical(df.subject)
    df['label'] = df.subject.cat.codes
    df.to_csv('./data/img_info.csv', index= False)
    print(f'Saved the csv to ./data/img_info.csv ')

if __name__ == '__main__':
    write_csv('./data/CASIA/*/*')