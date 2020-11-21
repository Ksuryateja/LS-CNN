from facenet_pytorch import MTCNN
from PIL import Image
import torch
import glob
from tqdm import tqdm
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def face_detection(destination_path = "./data/Faces-CASIA/",pattern = "/content/CASIA-maxpy-clean/*/*"):
    destination_path = "./data/CASIA/"
    pattern = "./data/CASIA-maxpy-clean/*/*" 
    mtcnn = MTCNN(post_process=False, device = device, image_size = 144) 
    undetected = 0
    for img in tqdm(glob.glob(pattern)):
        copy_path = destination_path + img.split('/')[3]
        if not os.path.exists(copy_path):
            os.mkdir(copy_path)    
        img_name = img.split('/')[4]
        img = Image.open(img)
        try:
            face = mtcnn(img, save_path = copy_path + '/' + img_name)
        except :
            undetected += 1
    print(f'Faces were undetected in {undetected} images') 