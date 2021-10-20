import cv2
import os
import numpy as np
#from PIL import Image
import matplotlib.pyplot as plt
#%matplotlib inline
import argparse

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def split_yanqd_mask_area(eyelid_mask_dir,cornea_mask_dir,cataract_mask_dir,save_file=None):
    """
    these mask from https://github.com/milesial/Pytorch-UNet model predictions
    """
    colors=[(0.0, 1.0, 0.40000000000000036),(1.0, 0.0, 0.0),(0.8000000000000007, 0.0, 1.0),(0.0, 0.40000000000000036, 1.0),(0.7999999999999998, 1.0, 0.0)]
    cataract_mask = cv2.imread(cataract_mask_dir)
    if len(cataract_mask.shape) == 3:
        cataract_mask=cv2.cvtColor(cataract_mask,cv2.COLOR_BGR2GRAY)
    
    cornea_mask =cv2.imread(cornea_mask_dir)
    if len(cornea_mask.shape) ==3:
        cornea_mask=cv2.cvtColor(cornea_mask,cv2.COLOR_BGR2GRAY)
    
    eyelid_mask  = cv2.imread(eyelid_mask_dir)
    if len(eyelid_mask.shape)==2:
        eyelid_mask=np.concatenate([eyelid_mask_gray[:,:,np.newaxis],eyelid_mask_gray[:,:,np.newaxis],eyelid_mask_gray[:,:,np.newaxis]],axis=2)
    #eyelid_mask_gray = cv2.cvtColor(eyelid_mask,cv2.COLOR_BGR2GRAY)
    cornea_mask_2=cornea_mask.copy()
    cornea_mask_2[cornea_mask_2>1]=1

    cataract_mask_1=cataract_mask.copy()
    cataract_mask_1[cataract_mask_1>1]=1

    img_mask = apply_mask(eyelid_mask,cornea_mask_2,colors[0])
    img_mask = apply_mask(img_mask,cataract_mask_1,colors[1])
    if save_file is not None:
        plt.imsave(save_file,img_mask)
    return img_mask


if  __name__=="__main__":
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--outfile',type=str, default="test.png", help='outfile')
    parser.add_argument("--eyelid_mask_dir",type=str,default=None,help="eyelid_mask_dir")

    parser.add_argument("--cornea_mask_dir",type=str,default=None,help="cornea_mask_dir")
    parser.add_argument("--cataract_mask_dir",type=str,default=None,help="cataract_mask_dir")

    args=parser.parse_args()

    _=split_yanqd_mask_area(args.eyelid_mask_dir,args.cornea_mask_dir,args.cataract_mask_dir,args.outfile)

