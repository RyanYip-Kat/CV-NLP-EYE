import os
import cv2
import numpy as np
import argparse

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def remove_mask(image,mask,use_green=False):
    assert len(mask.shape)==2
    if len(image.shape)==3:
        if use_green:
            r,g,b=cv2.split(image)
            img_gray = g.copy()
        else:
            img_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image.copy()

    img_s = np.subtract(img_gray,mask)
    return img_s


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir",type=str,default=None)
    parser.add_argument("--image_dir",type=str,default=None)
    parser.add_argument("--outdir",type=str,default="image_no_mask")
    parser.add_argument("--img_size",type=int,default=None)
    args = parser.parse_args()

    image_dir = args.image_dir
    mask_dir  = args.mask_dir
    outdir = args.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    imgs = [x for x in os.listdir(image_dir) if "jpg" in x]
    masks = [x.split(".")[0]+"_pred.png"  for x in imgs]
    for i in tqdm(range(len(imgs)),total=len(imgs)):
        img = imgs[i]
        mask = masks[i]
        filename = outdir +"/" + img.split(".")[0]+"_mask.png"
        img_file = image_dir +"/" + img
        mask_file = mask_dir +"/" + mask


        im = Image.open(img_file)
        wi,hi=im.size
        mim = Image.open(mask_file).convert("L")
        wm,hm=mim.size
        if args.img_size is None:
            w = min(wi,wm)
            h=  min(hi,hm)
            mim = mim.resize((w,h))
            im = im.resize((w,h))
        else:
            mim = mim.resize((args.img_size,args.img_size))
            im = im.resize((args.img_size,args.img_size))
        
        im_arr= np.array(im)
        mim_arr = np.array(mim)

        img_s = remove_mask(im_arr,mim_arr,True)
        plt.imsave(filename,img_s,cmap="gray")

    
