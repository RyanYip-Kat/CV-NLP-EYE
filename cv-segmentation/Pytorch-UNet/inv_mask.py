import cv2
import os
import numpy as np
from tqdm import tqdm
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
#%matplotlib inline
import argparse


if  __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_dir',type=str, default=None, help='mask dir ')
    parser.add_argument('--outdir',type=str, default="./inv_mask", help='outfile')
    args=parser.parse_args()

    maskdir = args.mask_dir
    outdir  = args.outdir
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    imgs =os.listdir(maskdir)
    for img  in tqdm(imgs,total=len(imgs)):
        mask  = Image.open(maskdir+"/"+img).convert("L")
        mask = np.asarray(mask,dtype=np.float32)

        inv_mask = 1-mask 
        filename =  outdir +"/"+img.split(".")[0]+"_inv.png"
        cv2.imwrite(filename,inv_mask)
