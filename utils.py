from __future__ import print_function
import glob
import pandas as pd
import re

import os
import logging

import visdom
import numpy as np
import time

class Visualizer():
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = 1

    def plot_curves(self, d, iters, title='loss', xlabel='iters', ylabel='accuracy'):
        name = list(d.keys())
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        self.vis.line(Y=y,
                      X=np.array([self.index]),
                      win=title,
                      opts=dict(legend=name, title = title, xlabel=xlabel, ylabel=ylabel),
                      update=None if self.index == 0 else 'append')
        self.index = iters

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
    write_csv('./data/CASIA-washed/*/*')